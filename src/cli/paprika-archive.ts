import { Command } from "commander";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { gunzipSync, inflateRawSync } from "node:zlib";

function loadDotEnv(file = ".env"): void {
  if (!existsSync(file)) return;

  for (const line of readFileSync(file, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const separator = trimmed.indexOf("=");
    if (separator === -1) continue;

    const key = trimmed.slice(0, separator).trim();
    const value = trimmed.slice(separator + 1).trim().replace(/^"(.*)"$/, "$1");

    if (!(key in process.env)) {
      process.env[key] = value;
    }
  }
}

interface GeminiClientOptions {
  apiKey: string;
  model?: string;
  baseUrl?: string;
}

interface GeminiJsonRequest {
  prompt: string;
  schemaHint?: string;
}

class GeminiClient {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;

  constructor(options: GeminiClientOptions) {
    this.apiKey = options.apiKey;
    this.model = options.model ?? "gemini-3.5-flash-lite";
    this.baseUrl = options.baseUrl ?? "https://generativelanguage.googleapis.com/v1beta";
  }

  async generateJson<T>(request: GeminiJsonRequest): Promise<T> {
    const prompt = request.schemaHint
      ? `${request.prompt}\n\nReturn only valid JSON matching this shape:\n${request.schemaHint}`
      : request.prompt;

    const response = await fetch(
      `${this.baseUrl}/models/${this.model}:generateContent?key=${encodeURIComponent(this.apiKey)}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [
                {
                  text: prompt
                }
              ]
            }
          ],
          generationConfig: {
            responseMimeType: "application/json",
            temperature: 0.4
          }
        })
      }
    );

    if (!response.ok) {
      throw new Error(`Gemini HTTP ${response.status}: ${await response.text()}`);
    }

    const payload = (await response.json()) as {
      candidates?: Array<{
        content?: {
          parts?: Array<{
            text?: string;
          }>;
        };
      }>;
    };

    const text = payload.candidates?.[0]?.content?.parts
      ?.map((part) => part.text ?? "")
      .join("") ?? "";

    return JSON.parse(text) as T;
  }
}

interface PaprikaRecipe {
  sourceEntry: string;
  title: string;
  description: string;
  servings?: number;
  ingredients: string[];
  instructions: string[];
  notes?: string;
  categories: string[];
  imageUrl?: string;
}

interface ZipEntry {
  name: string;
  data: Buffer;
  compressionMethod: number;
}

function readZipEntries(buffer: Buffer): ZipEntry[] {
  let eocdOffset = -1;

  for (let index = buffer.length - 22; index >= 0; index -= 1) {
    if (buffer.readUInt32LE(index) === 0x06054b50) {
      eocdOffset = index;
      break;
    }
  }

  if (eocdOffset === -1) {
    throw new Error("File is not a ZIP archive.");
  }

  const entryCount = buffer.readUInt16LE(eocdOffset + 10);
  let centralDirectoryOffset = buffer.readUInt32LE(eocdOffset + 16);
  const entries: ZipEntry[] = [];

  for (let index = 0; index < entryCount; index += 1) {
    if (buffer.readUInt32LE(centralDirectoryOffset) !== 0x02014b50) {
      break;
    }

    const compressionMethod = buffer.readUInt16LE(centralDirectoryOffset + 10);
    const compressedSize = buffer.readUInt32LE(centralDirectoryOffset + 20);
    const nameLength = buffer.readUInt16LE(centralDirectoryOffset + 28);
    const extraLength = buffer.readUInt16LE(centralDirectoryOffset + 30);
    const commentLength = buffer.readUInt16LE(centralDirectoryOffset + 32);
    const localHeaderOffset = buffer.readUInt32LE(centralDirectoryOffset + 42);

    const name = buffer.toString(
      "utf8",
      centralDirectoryOffset + 46,
      centralDirectoryOffset + 46 + nameLength
    );

    const localNameLength = buffer.readUInt16LE(localHeaderOffset + 26);
    const localExtraLength = buffer.readUInt16LE(localHeaderOffset + 28);
    const dataStart = localHeaderOffset + 30 + localNameLength + localExtraLength;

    entries.push({
      name,
      data: buffer.subarray(dataStart, dataStart + compressedSize),
      compressionMethod
    });

    centralDirectoryOffset += 46 + nameLength + extraLength + commentLength;
  }

  return entries;
}

function splitLines(value: unknown): string[] {
  return String(value ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function mapPaprikaRecipe(
  recipe: Record<string, unknown>,
  sourceEntry: string
): PaprikaRecipe {
  const servingsText = String(recipe.servings ?? "");
  const servingsMatch = servingsText.match(/\d+/);
  const categories = Array.isArray(recipe.categories)
    ? recipe.categories.map(String)
    : [];

  return {
    sourceEntry,
    title: String(recipe.name ?? ""),
    description: String(recipe.description ?? ""),
    servings: servingsMatch ? Number(servingsMatch[0]) : undefined,
    ingredients: splitLines(recipe.ingredients),
    instructions: splitLines(recipe.directions),
    notes: String(recipe.notes ?? ""),
    categories,
    imageUrl: typeof recipe.image_url === "string" ? recipe.image_url : undefined
  };
}

function parsePaprikaArchive(filePath: string): PaprikaRecipe[] {
  const archive = readFileSync(filePath);
  const entries = readZipEntries(archive);
  const recipes: PaprikaRecipe[] = [];

  for (const entry of entries) {
    if (!entry.name.endsWith(".paprikarecipe")) continue;

    let entryData = entry.data;

    if (entry.compressionMethod === 8) {
      entryData = inflateRawSync(entryData);
    }

    const decompressed = entryData.subarray(0, 2).equals(Buffer.from([0x1f, 0x8b]))
      ? gunzipSync(entryData)
      : entryData;

    const recipe = JSON.parse(decompressed.toString("utf8")) as Record<string, unknown>;
    recipes.push(mapPaprikaRecipe(recipe, entry.name));
  }

  return recipes;
}

const program = new Command();

program
  .name("paprika-archive")
  .description("Convert Paprika archives and generate recipes with Gemini");

program
  .command("convert")
  .argument("<archive>", "Paprika .paprikarecipes export file")
  .option("--output <dir>", "Output directory", "converted")
  .action(async (archive: string, options: { output?: string }) => {
    const outputDir = options.output ?? "converted";
    await mkdir(outputDir, { recursive: true });

    const recipes = parsePaprikaArchive(archive);

    for (const recipe of recipes) {
      const slug = recipe.title
        .replace(/[^a-z0-9-_]+/gi, "-")
        .replace(/^-+|-+$/g, "")
        .toLowerCase() || "recipe";

      const outputFile = path.join(outputDir, `${slug}.json`);
      await writeFile(outputFile, `${JSON.stringify(recipe, null, 2)}\n`);
      console.log(`Converted: ${recipe.title} -> ${outputFile}`);
    }

    console.log(`Converted ${recipes.length} recipe(s).`);
  });

program
  .command("generate")
  .requiredOption("--ingredients <items>", "Comma-separated ingredients")
  .option("--servings <number>", "Number of servings", "4")
  .option("--cuisine <cuisine>", "Preferred cuisine")
  .option("--diet <diet>", "Dietary preference")
  .option("--model <model>", "Gemini model override")
  .option("--output <file>", "Output JSON file", "generated-recipe.json")
  .action(async (options: {
    ingredients: string;
    servings?: string;
    cuisine?: string;
    diet?: string;
    model?: string;
    output?: string;
  }) => {
    loadDotEnv();

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not set. Add it to .env or your environment.");
    }

    const client = new GeminiClient({
      apiKey,
      model: options.model ?? process.env.GEMINI_MODEL_GENERATE ?? "gemini-3.8-flash"
    });

    const prompt = [
      `Create an original recipe using these ingredients: ${options.ingredients}.`,
      `Servings: ${options.servings ?? "4"}.`,
      options.cuisine ? `Preferred cuisine: ${options.cuisine}.` : "",
      options.diet ? `Dietary preference: ${options.diet}.` : "",
      "Use clear numbered steps and realistic ingredient quantities.",
      "Do not include markdown, explanations, or JSON outside the requested JSON object."
    ].filter(Boolean).join("\n");

    const recipe = await client.generateJson<Record<string, unknown>>({
      prompt,
      schemaHint: `{\n  "title": string,\n  "description": string,\n  "servings": number,\n  "ingredients": string[],\n  "steps": string[],\n  "notes": string\n}`
    });

    const output = options.output ?? "generated-recipe.json";
    await writeFile(output, `${JSON.stringify(recipe, null, 2)}\n`);
    console.log(`Saved recipe to ${output}`);
  });

await program.parseAsync(process.argv);
