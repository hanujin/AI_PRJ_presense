export const presentationRecords = [
  {
    id: "PS-024",
    title: "Capstone Midterm Rehearsal",
    date: "2026-05-29",
    duration: "08m 42s",
    result: "Moderate pressure",
    stressAverage: "0.54",
    diagnosis: "Transitions became rushed during the model explanation section.",
  },
  {
    id: "PS-023",
    title: "Mock Interview Practice",
    date: "2026-05-26",
    duration: "06m 15s",
    result: "Mostly stable",
    stressAverage: "0.42",
    diagnosis: "Calm delivery overall, but eye focus likely weakened near the closing answer.",
  },
  {
    id: "PS-022",
    title: "Final Demo Script Run",
    date: "2026-05-21",
    duration: "09m 03s",
    result: "Noticeable peak",
    stressAverage: "0.61",
    diagnosis: "Technical terms stacked too quickly without pause support, increasing perceived tension.",
  },
];

export const diagnosisHistory = [
  {
    label: "Most common issue",
    value: "Rushed transitions",
    detail: "Detected in 2 of the last 3 sessions.",
  },
  {
    label: "Best recent improvement",
    value: "Opening stability",
    detail: "Your opening segment looks calmer and more consistent than before.",
  },
  {
    label: "Next coaching focus",
    value: "Technical explanation pacing",
    detail: "Pause one beat after each key term or metric.",
  },
];

export type SettingToggle = {
  id: string;
  label: string;
  enabled: boolean;
};

export type SettingsSection = {
  title: string;
  items: SettingToggle[];
};

export const primarySettings: SettingToggle[] = [
  {
    id: "camera-preview",
    label: "Default to live camera preview",
    enabled: true,
  },
  {
    id: "microphone-active-only",
    label: "Only capture during active sessions",
    enabled: true,
  },
  {
    id: "session-summary-notice",
    label: "Show summary after each session",
    enabled: true,
  },
];

export const settingsSections: SettingsSection[] = [
  {
    title: "Session Preferences",
    items: [
      {
        id: "default-camera-view",
        label: "Default camera view on practice start",
        enabled: true,
      },
      {
        id: "mute-live-feedback",
        label: "Mute live feedback until session ends",
        enabled: false,
      },
      {
        id: "auto-ai-summary",
        label: "Generate post-session AI coaching summary automatically",
        enabled: true,
      },
    ],
  },
  {
    title: "Privacy Controls",
    items: [
      {
        id: "store-summaries-only",
        label: "Store session summaries only",
        enabled: true,
      },
      {
        id: "disable-frame-storage",
        label: "Do not save raw camera preview frames",
        enabled: true,
      },
      {
        id: "mic-active-practice-only",
        label: "Allow microphone access only during active practice",
        enabled: true,
      },
    ],
  },
  {
    title: "AI Agent Preferences",
    items: [
      {
        id: "concise-coaching-tone",
        label: "Use concise coaching tone",
        enabled: true,
      },
      {
        id: "prioritize-presentation-advice",
        label: "Prioritize presentation-specific advice",
        enabled: true,
      },
      {
        id: "future-api-key-connection",
        label: "Prepare for future API key connection",
        enabled: false,
      },
    ],
  },
];

export function buildDefaultPrimarySettings() {
  return Object.fromEntries(primarySettings.map((item) => [item.id, item.enabled])) as Record<string, boolean>;
}

export function buildDefaultSectionSettings() {
  return Object.fromEntries(
    settingsSections.flatMap((section) => section.items.map((item) => [item.id, item.enabled])),
  ) as Record<string, boolean>;
}
