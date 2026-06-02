export const initialTimeline = [28, 34, 31, 42, 48, 52, 46, 58, 49, 44, 38, 36];

export const postSessionHighlights = [
  {
    title: "Opening Stability",
    detail: "The first segment stayed controlled, which means your opening script is already a strong anchor.",
  },
  {
    title: "Technical Explanation Pressure",
    detail: "Stress likely rose while explaining detailed content too quickly without enough transition language.",
  },
  {
    title: "Best Improvement Lever",
    detail: "Short pauses between slide changes will probably improve both clarity and perceived confidence.",
  },
];

export const agentStarters = [
  "Which part of my presentation seems most stressful?",
  "What should I improve before the next rehearsal?",
  "How can I sound calmer during technical explanations?",
];

export const practiceScenes = [
  {
    id: "camera",
    label: "Live Camera",
    description: "Use your real webcam preview during practice.",
  },
  {
    id: "audience",
    label: "Audience Mock",
    description: "Simulate a small presentation audience scene.",
  },
  {
    id: "interview",
    label: "Interview Panel",
    description: "Practice with a mock interview-style visual.",
  },
  {
    id: "slides",
    label: "Presentation Slides",
    description: "Focus with a clean slide-style practice backdrop.",
  },
] as const;
