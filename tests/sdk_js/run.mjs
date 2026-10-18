// Runs TypeSafe's JavaScript SDK against a Jez server. Configuration comes only from
// TYPESAFE_API_KEY, TYPESAFE_BASE_URL and TYPESAFE_DEFAULT_MODEL. Prints one JSON report.
//
//   node run.mjs <examples-dir>
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { TypeSafeClient, noul, choice, score } from "@typesafe-ai/sdk";

const dir = process.argv[2];
const report = { examples: {}, errors: {}, models: null, typed: null };
const noRetry = { maxRetries: 0 };

const client = new TypeSafeClient({ retry: noRetry });

for (const file of readdirSync(dir).filter((f) => f.endsWith(".json")).sort()) {
  const req = JSON.parse(readFileSync(join(dir, file), "utf8"));
  try {
    const res = await client.systemOne({ state: req.state, questions: req.questions }); // model from env
    const ok = res.model === process.env.JEZ_EXPECT_MODEL && Object.keys(res.answers).sort().join() === Object.keys(req.questions).sort().join();
    report.examples[file] = { ok, model: res.model, usage: res.usage, answers: res.answers };
  } catch (e) {
    report.examples[file] = { ok: false, error: `${e.constructor.name}: ${e.message}` };
  }
}

try {
  const res = await client.systemOne({
    state: { message: "I was charged twice. Please help." },
    questions: {
      billing: noul("Is this about billing?"),
      tone: choice("What is the tone?", { calm: null, angry: null }),
      urgency: score("How urgent?", ["Can wait", "This week", "Today"]),
    },
  });
  report.typed = { ok: res.answers.billing.type === "noul" && res.answers.tone.type === "choice" && res.answers.urgency.type === "score", answers: res.answers };
} catch (e) {
  report.typed = { ok: false, error: `${e.constructor.name}: ${e.message}` };
}

try {
  const models = await client.models.list();
  report.models = (models.models ?? models).map((m) => m.name).sort();
} catch (e) {
  report.models = { error: `${e.constructor.name}: ${e.message}` };
}

const errorCases = {
  400: () => {
    let nested = "x";
    for (let i = 0; i < 70; i++) nested = [nested];
    return client.systemOne({ state: nested, questions: { q: noul("deep?") } });
  },
  401: () => new TypeSafeClient({ apiKey: "wrong-key", retry: noRetry }).systemOne({ state: "x", questions: { q: noul("y") } }),
  413: () => client.systemOne({ state: "a".repeat(32 * 1024 * 1024 + 64), questions: { q: noul("big?") } }, { timeout: 120000 }),
  // The SDK owns the Content-Encoding header, so a custom fetch rewrites it to an encoding Jez rejects.
  415: () => new TypeSafeClient({
    retry: noRetry,
    compression: false,
    fetch: (url, init) => {
      const headers = new Headers(init?.headers);
      headers.set("Content-Encoding", "br");
      return fetch(url, { ...init, headers });
    },
  }).systemOne({ state: "x", questions: { q: noul("y") } }),
  422: () => client.systemOne({ state: "x", questions: { q: { type: "choice", instructions: "y", criteria: { only: null } } } }),
  "422-jev": () => client.systemOne({ state: "x", questions: { q: noul("y") }, model: "jev-latest" }),
  529: () => client.systemOne({ state: "x", questions: { q: noul("y") }, model: "jez-latest" }),
};
if (process.env.JEZ_RATE_LIMITED_URL) {
  errorCases[429] = () => new TypeSafeClient({ baseURL: process.env.JEZ_RATE_LIMITED_URL, retry: { maxRetries: 1, backoffInitial: 100, backoffMax: 200 } })
    .systemOne({ state: "x", questions: { q: noul("y") } });
}
for (const [code, run] of Object.entries(errorCases)) {
  try {
    await run();
    report.errors[code] = { raised: null };
  } catch (e) {
    report.errors[code] = { raised: e.constructor.name, status: e.status ?? null, retryAfterMs: e.retryAfterMs ?? null };
  }
}

console.log(JSON.stringify(report));
