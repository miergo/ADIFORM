import { assetUrl } from "./assetUrl.js";

export const SOUNDS = {
  counter: assetUrl("audio/counter.mp3"),
  multiplier: assetUrl("audio/multiplierSoundEffect.mp3"),
  score: assetUrl("audio/scorepage_sound.mp3"),
  background: assetUrl("audio/background_theme.mp3"),
  buttonHover: assetUrl("audio/buttonHoversound.mp3"),
  sessionThemes: [
    assetUrl("audio/WebCamUploadThemeMusic/ThemeSong1.mp3"),
    assetUrl("audio/WebCamUploadThemeMusic/ThemeSong2.mp3"),
  ],
};

const VOLUME = {
  counter: 0.4,
  multiplier: 0.4,
  score: 0.5,
  background: 0.7,
  buttonHover: 0.5,
  sessionThemes: 0.7,
}

let music = null;
let unlocked = false;

export function playSfx(key) {
  const src = SOUNDS[key];
  if (!src) {
    return null;
  }
  const audio = new Audio(src);
  audio.volume = VOLUME[key];
  audio.play().catch(() => { });
  return audio;
}

export function stopSfx(audio) {
  if (!audio) {
    return;
  }
  audio.pause();
  audio.currentTime = 0;
}

export function playMusic(src, { muted = false } = {}) {
  if (!src) {
    return;
  }
  const startMuted = muted && !unlocked;
  if (music?.src.endsWith(src)) {
    music.muted = startMuted;
    if (music.paused) {
      music.play().catch(() => { });
    }
    return;
  }
  stopMusic();
  music = new Audio(src);
  music.volume =
    src === SOUNDS.background ? VOLUME.background : VOLUME.sessionThemes;
  music.loop = true;
  music.muted = startMuted;
  music.play().catch(() => { });
}

export function unmuteMusic() {
  unlocked = true;
  if (!music) {
    return;
  }
  music.muted = false;
  if (music.paused) {
    music.play().catch(() => { });
  }
}

export function stopMusic() {
  if (!music) {
    return;
  }
  music.pause();
  music.currentTime = 0;
  music = null;
}

export function randomSessionTheme() {
  const themes = SOUNDS.sessionThemes;
  return themes[Math.floor(Math.random() * themes.length)];
}
