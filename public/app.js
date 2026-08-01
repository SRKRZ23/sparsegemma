// PocketTriage — fully on-device health-navigation companion.
// Everything below runs client-side in the browser via WebGPU. No network calls happen
// after the one-time model download from the Hugging Face CDN — no user input is ever
// sent anywhere. Built for the "On-Device Private Health" track.
import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/+esm";

const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";

const statusEl = document.getElementById("status");
const loadfill = document.getElementById("loadfill");
const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

addMsg("sys", "Loading Gemma 4 (E2B, q4f16) — first load downloads the model once; it's cached afterward.");

const SYSTEM_PROMPT = `You are PocketTriage, a private, on-device health-navigation assistant. You are NOT a
doctor and you do NOT diagnose. Your job: (1) explain, in plain language, the general URGENCY category a
description of symptoms would typically fall under (similar to how ED triage systems like ESI/qSOFA reason
about urgency — routine / soon / urgent / emergency), (2) explain WHY in plain terms, (3) suggest 2-3
specific questions the person should ask a real clinician, and (4) always end by reminding them this is
educational only and to seek real medical care, calling emergency services immediately for anything
severe (chest pain, difficulty breathing, stroke signs, severe bleeding, loss of consciousness). Keep
answers concise (under 180 words). Never ask for or reference real identifying personal information.`;

let model, processor;
let history = [{ role: "system", content: SYSTEM_PROMPT }];

async function init() {
  try {
    processor = await AutoProcessor.from_pretrained(MODEL_ID);
    model = await Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: "q4f16",
      device: "webgpu",
      progress_callback: (info) => {
        if (info.status === "progress" && info.progress != null) {
          loadfill.style.width = `${Math.round(info.progress)}%`;
        }
      },
    });
    statusEl.textContent = "ready — 100% on-device";
    statusEl.classList.remove("off");
    statusEl.classList.add("on");
    sendBtn.disabled = false;
    loadfill.style.width = "100%";
    addMsg("sys", "Model loaded. Nothing you type is sent anywhere — inference runs locally on your GPU via WebGPU.");
  } catch (err) {
    statusEl.textContent = "failed to load (needs a WebGPU browser, e.g. recent Chrome/Edge)";
    addMsg("sys", `Load error: ${err?.message || err}`);
    console.error(err);
  }
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || !model) return;
  inputEl.value = "";
  sendBtn.disabled = true;
  addMsg("user", text);
  history.push({ role: "user", content: text });

  const botDiv = addMsg("bot", "");
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      botDiv.textContent += token;
      chatEl.scrollTop = chatEl.scrollHeight;
    },
  });

  try {
    const inputs = await processor.apply_chat_template(history, {
      add_generation_prompt: true,
      return_dict: true,
    });
    const output = await model.generate({
      ...inputs,
      max_new_tokens: 320,
      do_sample: false,
      streamer,
    });
    const full = botDiv.textContent;
    history.push({ role: "assistant", content: full });
  } catch (err) {
    botDiv.textContent = `(generation error: ${err?.message || err})`;
    console.error(err);
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

init();
