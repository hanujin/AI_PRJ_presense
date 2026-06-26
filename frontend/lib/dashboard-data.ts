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

export type PresentationSlide =
  | { kind: "title"; eyebrow: string; title: string; subtitle: string }
  | { kind?: "content"; title: string; bullets: string[] };

export const samplePresentation: { name: string; slides: PresentationSlide[] } = {
  name: "PreSense Presentation",
  slides: [
    {
      kind: "title",
      eyebrow: "Graduation Project",
      title: "PreSense",
      subtitle: "Real-time presentation anxiety sensing & coaching",
    },
    {
      title: "The Problem",
      bullets: [
        "Most students feel intense anxiety during live presentations.",
        "Stress is hard to notice and correct while you are speaking.",
        "Existing tools give feedback only after the talk is over.",
      ],
    },
    {
      title: "Our Approach",
      bullets: [
        "Sense stress in real time from camera and microphone signals.",
        "Give calm, non-distracting cues during the rehearsal.",
        "Summarize pressure points and next steps after the session.",
      ],
    },
    {
      title: "How It Works",
      bullets: [
        "Webcam + mic stream feeds the on-device student model.",
        "The stress score updates continuously during your talk.",
        "A post-session report highlights peak-pressure moments.",
      ],
    },
    {
      title: "Why It Matters",
      bullets: [
        "Practice in realistic classroom and interview scenes.",
        "Build steadier pacing before the real presentation.",
        "Turn anxiety into measurable, improvable feedback.",
      ],
    },
    {
      kind: "title",
      eyebrow: "Thank You",
      title: "Questions?",
      subtitle: "PreSense — present with confidence",
    },
  ],
};

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
    visual: { type: "camera" },
  },
  {
    id: "audience",
    label: "Classroom Audience",
    description: "Practice with a classroom audience looking toward you.",
    visual: { type: "image", src: "/scenes/classroom-audience.png" },
  },
  {
    id: "interview",
    label: "Interview Panel",
    description: "Practice with a calm mock interview panel.",
    visual: { type: "panel" },
  },
  {
    id: "slides",
    label: "Presentation Slides",
    description: "Show your uploaded deck as the main rehearsal screen.",
    visual: { type: "deck" },
  },
  {
    id: "classroom-slides",
    label: "Slides in Classroom",
    description: "Practice with your deck in front of a classroom audience.",
    visual: { type: "classroomDeck", src: "/scenes/classroom-audience.png" },
  },
] as const;
