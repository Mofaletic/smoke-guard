import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { startServer } from "./server.mjs";

const ROOT = resolve(fileURLToPath(new URL(".", import.meta.url)));
const FACE_SERVICE_URL = process.env.FACE_SERVICE_URL || "http://127.0.0.1:8787";
const HEALTH_URL = `${FACE_SERVICE_URL}/health`;
const START_FACE_SCRIPT = join(ROOT, "scripts", "start_face_service.ps1");

let faceProcess = null;
let dashboard = null;

async function canUseFaceScript() {
  try {
    await access(START_FACE_SCRIPT, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function isFaceServiceReady() {
  try {
    const response = await fetch(HEALTH_URL);
    return response.ok;
  } catch {
    return false;
  }
}

async function ensureFaceService() {
  if (await isFaceServiceReady()) {
    console.log(`Face service already running: ${FACE_SERVICE_URL}`);
    return;
  }

  if (!(await canUseFaceScript())) {
    console.warn(`Face service script not found: ${START_FACE_SCRIPT}`);
    return;
  }

  console.log("Starting bundled Java face service for D5 project ...");
  faceProcess = spawn(
    "powershell",
    ["-ExecutionPolicy", "Bypass", "-File", START_FACE_SCRIPT],
    {
      cwd: ROOT,
      stdio: "inherit",
      windowsHide: false,
    },
  );

  faceProcess.on("exit", (code, signal) => {
    const reason = signal ? `signal ${signal}` : `code ${code}`;
    console.log(`Face service process exited with ${reason}`);
  });

  for (let attempt = 0; attempt < 72; attempt += 1) {
    await delay(5000);
    if (await isFaceServiceReady()) {
      console.log(`Face service ready: ${FACE_SERVICE_URL}`);
      return;
    }
  }

  console.warn("Face service did not become ready within 6 minutes. Dashboard will still start.");
}

function registerShutdown() {
  const shutdown = () => {
    if (dashboard) {
      dashboard.close();
    }
    if (faceProcess && !faceProcess.killed) {
      faceProcess.kill();
    }
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

async function main() {
  registerShutdown();
  await ensureFaceService();
  dashboard = startServer();
}

main().catch((error) => {
  console.error("Failed to start D5 project:", error);
  process.exit(1);
});
