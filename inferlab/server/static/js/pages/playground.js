import { stream } from "../api.js";
import { attachTip, clear, fmt, h, ms } from "../dom.js";
import { columns } from "../charts.js";
import { area, card, check, code, field, message, num, page, seg, slider, stats, text } from "../ui.js";

let controller = null;

export default {
  id: "playground",
  title: "Playground",
  teardown() {
    if (controller) controller.abort();
  },
  render(ctx) {
    const chat = ctx.overview.model.chat_template === "llama2";
    const prompt = area(chat ? "Write a haiku about GPUs." : "Once upon a time, there was a little robot who", 4);
    const maxTokens = num(96, 1, ctx.overview.security.max_tokens);
    const temp = slider("Temperature", 0.8, 0, 2, 0.05, (v) => v.toFixed(2));
    const topP = slider("Top-p", 0.95, 0.05, 1, 0.05, (v) => v.toFixed(2));
    const topK = num(0, 0, 1000);
    const stop = text("", "stop string (optional)");
    const seed = num("", 0, 2 ** 31);
    seed.placeholder = "random";
    const showTok = check("show tokens", true);
    const mode = seg([["completion", "Completion"], ["chat", "Chat"]], chat ? "chat" : "completion", () => {});

    const out = h("div", { class: "output", "aria-live": "polite" });
    const tokenBox = h("div", { class: "tokens" });
    const statBox = h("div");
    const chartBox = h("div");
    const err = h("div");
    const curl = h("div");

    const go = h("button", { type: "button", class: "primary", text: "Generate" });
    const stopBtn = h("button", { type: "button", text: "Stop", disabled: true });
    stopBtn.addEventListener("click", () => controller && controller.abort());

    go.addEventListener("click", async () => {
      clear(err);
      clear(out);
      clear(tokenBox);
      clear(chartBox);
      const isChat = mode.get() === "chat";
      const body = {
        max_tokens: Number(maxTokens.value) || 64,
        temperature: Number(temp.input.value),
        top_p: Number(topP.input.value),
        top_k: Number(topK.value) || 0,
        stream: true,
      };
      if (stop.value) body.stop = [stop.value];
      if (seed.value !== "") body.seed = Number(seed.value);
      let path = "/v1/completions";
      if (isChat) {
        path = "/v1/chat/completions";
        body.messages = [{ role: "user", content: prompt.value }];
      } else {
        body.prompt = prompt.value;
        body.logprobs = 5;
        out.appendChild(h("span", { class: "prompt", text: prompt.value }));
      }
      const gen = h("span", { class: "gen" });
      out.appendChild(gen);
      curl.replaceChildren(code(
        `curl -N http://${location.host}${path} \\\n  -H "Authorization: Bearer $INFERLAB_API_KEY" ` +
        `-H "Content-Type: application/json" \\\n  -d '${JSON.stringify(body)}'`));

      controller = new AbortController();
      go.disabled = true;
      stopBtn.disabled = false;
      const t0 = performance.now();
      let tFirst = null;
      let tLast = t0;
      const gaps = [];
      let n = 0;
      let serverMetrics = null;
      try {
        await stream(path, body, (ev, t) => {
          const c = ev.choices && ev.choices[0];
          if (ev.x_metrics) serverMetrics = ev.x_metrics;
          if (!c) return;
          const piece = c.text !== undefined ? c.text : (c.delta && c.delta.content) || "";
          if (!piece && !(c.logprobs && c.logprobs.tokens && c.logprobs.tokens.length)) return;
          const dt = t - tLast;
          if (tFirst === null) tFirst = t;
          else gaps.push(dt);
          tLast = t;
          n += 1;
          gen.textContent += piece;
          if (showTok.input.checked) {
            const lp = c.logprobs;
            const label = lp && lp.tokens ? lp.tokens[0] : piece;
            const chip = h("span", { class: "tok", text: label });
            let tip = `token ${n}  (+${fmt(dt, 1)} ms after the previous event)`;
            if (lp && lp.top_logprobs && lp.top_logprobs[0]) {
              const alts = Object.entries(lp.top_logprobs[0]).sort((a, b) => b[1] - a[1])
                .map(([tok, v]) => `${JSON.stringify(tok)}  p=${(Math.exp(v) * 100).toFixed(1)}%`);
              tip += `\nchosen logprob ${fmt(lp.token_logprobs[0], 3)}\ntop alternatives:\n${alts.join("\n")}`;
            }
            attachTip(chip, tip);
            tokenBox.appendChild(chip);
          }
        }, controller.signal);
      } catch (e) {
        if (e.name !== "AbortError") err.replaceChildren(message("error", e.message));
      } finally {
        go.disabled = false;
        stopBtn.disabled = true;
      }
      const total = tLast - t0;
      const ttft = tFirst === null ? null : tFirst - t0;
      const tpot = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : null;
      statBox.replaceChildren(stats([
        ["Time to first token", ms(ttft), "includes network + queue + prefill"],
        ["Time per output token", ms(tpot), "mean inter-token gap"],
        ["Throughput", n && total ? `${fmt((n / total) * 1000, 1)} tok/s` : "–", `${n} tokens`],
        ["Server queue time", serverMetrics ? ms(serverMetrics.queue_ms) : "–",
          serverMetrics && serverMetrics.cached_prompt_tokens ? `${serverMetrics.cached_prompt_tokens} prompt tokens from prefix cache` : "measured in the engine"],
      ]));
      if (gaps.length > 1) {
        chartBox.replaceChildren(h("h3", { text: "Inter-token latency (ms) - decode steps of the batched engine" }),
          columns({ labels: gaps.map((_, i) => i + 2), series: [{ name: "gap", cls: "c1", values: gaps }],
            valueFmt: (v) => fmt(v, 1), height: 160 }));
      }
    });

    return page("Production · try it", "Playground",
      "Streams from the OpenAI-compatible endpoint exactly like any client would. Hover a token to see its " +
      "probability and the alternatives the model considered.",
      h("div", { class: "grid cols-2" },
        card("Request", null,
          mode.el,
          field("Prompt", prompt),
          h("div", { class: "form" },
            field("Max tokens", maxTokens), temp.el, topP.el, field("Top-k (0 = off)", topK),
            field("Stop", stop), field("Seed", seed)),
          h("div", { class: "row" }, go, stopBtn, showTok.el),
          err, curl),
        card("Response", "Grey = your prompt, black = generated.", out, tokenBox, statBox, chartBox)));
  },
};
