// Video section of the project page.
// Each clip gets its video, a frame counter, and eight questions about the whole clip.
// A question box shows only the question until the video reaches the frame from which
// it can be answered. Then the answer appears and the box turns green. Seeking back
// hides the answers again, so the boxes always match the frame on screen.
"use strict";

const SPEEDS = [0.25, 0.5, 1];

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

const setText = (node, text) => { if (node.textContent !== text) node.textContent = text; };

function pill(person) {
  const tag = el("span", { class: "who" }, person.label);
  tag.style.setProperty("--c", person.color);
  return tag;
}

// Fills a node with text that has people marked in it. Each segment is [person id, text].
// A marked person gets their video colour and label, so two people are easy to tell apart.
function fillSegments(node, segments, people) {
  node.textContent = "";
  for (const [pid, text] of segments) {
    const person = pid && people[pid];
    if (!person) { node.append(text); continue; }
    const span = el("span", { class: "pname" }, el("b", {}, person.label));
    if (text.trim().toUpperCase() !== person.label) span.append(text);  // a bare "P2" is only the tag
    span.style.setProperty("--c", person.color);
    node.append(span);
  }
}

function buildClip(clip, n, total) {
  const N = clip.frames.length;
  const fps = clip.fps;
  const people = Object.fromEntries(clip.people.map((p) => [p.id, p]));
  const section = el("section", { class: "clip", id: `video-${n}` });

  section.append(
    el("h3", {}, `Video ${n} of ${total}`),
    el("p", { class: "meta" }, el("code", {}, clip.clip), ` · ${N} frames · ${clip.people.length} people`),
  );

  const video = el("video", {
    src: clip.video, muted: "", loop: "", playsinline: "", controls: "", preload: "metadata",
    width: clip.width, height: clip.height,
  });
  video.muted = true;
  section.append(video);

  // frame counter, single steps and playback speed
  const counter = el("span", { class: "counter" }, `Frame 1 / ${N}`);
  const answeredCount = el("span", { class: "answered-count" });
  const speedButtons = SPEEDS.map((s) =>
    el("button", { class: "btn-small" + (s === 1 ? " on" : ""), type: "button", onclick: () => setSpeed(s) }, `${s}×`));
  section.append(el("div", { class: "controls" },
    counter,
    el("span", { class: "group" },
      el("button", { class: "btn-small", type: "button", "aria-label": "Previous frame", onclick: () => step(-1) }, "◀ Frame"),
      el("button", { class: "btn-small", type: "button", "aria-label": "Next frame", onclick: () => step(1) }, "Frame ▶")),
    el("span", { class: "group" }, el("span", { class: "group-label" }, "Speed"), ...speedButtons),
    answeredCount,
  ));

  const indexAt = (t) => Math.min(N - 1, Math.max(0, Math.floor(t * fps + 0.01)));

  function seek(i) {
    if (i < 0) return;
    video.pause();
    video.currentTime = (i + 0.5) / fps;
    render(i);
  }
  function step(d) { seek(Math.min(N - 1, Math.max(0, indexAt(video.currentTime) + d))); }
  function setSpeed(s) {
    video.playbackRate = s;
    speedButtons.forEach((b, k) => b.classList.toggle("on", SPEEDS[k] === s));
  }

  // The boxes in one row of the grid get the height of the tallest head
  // (title, people and question), so that all answers start on the same line.
  function alignQuestions() {
    boxes.forEach((b) => { b.head.style.minHeight = ""; });
    const rows = new Map();
    boxes.forEach((b) => {
      const top = b.box.offsetTop;
      rows.set(top, Math.max(rows.get(top) || 0, b.head.offsetHeight));
    });
    boxes.forEach((b) => { b.head.style.minHeight = `${rows.get(b.box.offsetTop)}px`; });
  }

  const boxes = clip.temporal.map((t) => {
    const type = el("span", { class: "qtype" }, t.title);
    const chip = el("button", { class: "chip", type: "button", title: "Show the frame from which this can be answered" },
      `Frame ${t.reveal}`);
    chip.addEventListener("click", () => seek(clip.frames.indexOf(t.reveal)));
    const whos = t.person_ids.map((id) => people[id]).filter(Boolean).map(pill);
    const q = el("p", { class: "qtext" });
    fillSegments(q, t.q, people);
    const a = el("p", { class: "atext pending" }, `Waiting for frame ${t.reveal}…`);
    // the frames the answer rests on, in order. Each range jumps to its first frame.
    const evidence = t.evidence.length ? el("ol", { class: "evidence", hidden: "" }, ...t.evidence.map((e) => {
      const li = el("li");
      fillSegments(li, e.seg, people);
      if (e.start != null) {
        li.append(" ", el("button", { class: "evchip", type: "button", title: "Show this frame",
          onclick: () => seek(clip.frames.indexOf(e.start)) }, e.start === e.end ? `${e.start}` : `${e.start}-${e.end}`));
      }
      return li;
    })) : null;
    const head = el("div", { class: "qhead" }, type, el("div", { class: "qmeta" }, chip, ...whos), q);
    const box = el("div", { class: "q" }, head, a, evidence);
    return { box, head, a, evidence, t, shown: null };
  });
  section.append(el("div", { class: "qrow" }, ...boxes.map((b) => b.box)));

  function render(i) {
    const frame = clip.frames[i];
    counter.textContent = `Frame ${frame} / ${N}`;
    let count = 0;
    boxes.forEach((b) => {
      const on = frame >= b.t.reveal;
      if (on) count++;
      if (b.shown === on) return;
      b.shown = on;
      b.box.classList.toggle("answered", on);
      b.a.classList.toggle("pending", !on);
      if (on) fillSegments(b.a, b.t.a, people);
      else setText(b.a, `Waiting for frame ${b.t.reveal}…`);
      if (b.evidence) b.evidence.hidden = !on;
      if (on) {
        b.box.classList.remove("flash");
        void b.box.offsetWidth;  // restart the flash animation
        b.box.classList.add("flash");
      }
    });
    setText(answeredCount, `${count} of ${boxes.length} answered`);
  }

  // follow the video frame by frame when the browser can, otherwise use timeupdate
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const onFrame = (_, meta) => {
      render(Math.min(N - 1, Math.max(0, Math.round(meta.mediaTime * fps))));
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  } else {
    video.addEventListener("timeupdate", () => render(indexAt(video.currentTime)));
  }
  video.addEventListener("seeked", () => { if (video.paused) render(indexAt(video.currentTime)); });
  video.addEventListener("pause", () => {
    if (section.autoPaused) section.autoPaused = false;
    else section.userPaused = true;
    render(indexAt(video.currentTime));
  });
  video.addEventListener("play", () => { section.userPaused = false; });

  section.video = video;
  section.alignQuestions = alignQuestions;
  section.render = (i) => { render(i); alignQuestions(); };
  return section;
}

async function main() {
  const data = await (await fetch("data/clips.json")).json();
  const root = document.getElementById("clips");
  root.textContent = "";
  const sections = data.clips.map((c, k) => buildClip(c, k + 1, data.clips.length));
  sections.forEach((s) => root.append(s));
  sections.forEach((s) => s.render(0));

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => sections.forEach((s) => s.alignQuestions()), 200);
  });

  // Play the video that is on screen and pause the ones that scrolled away.
  // A video that the visitor paused stays paused.
  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      const s = v.closest(".clip");
      if (e.isIntersecting) {
        if (v.paused && !s.userPaused) v.play().catch(() => {});
      } else if (!v.paused) {
        s.autoPaused = true;
        v.pause();
      }
    }
  }, { threshold: 0.45 });
  sections.forEach((s) => observer.observe(s.video));
}

main().catch((err) => {
  document.getElementById("clips").textContent = `Could not load the videos: ${err}`;
});
