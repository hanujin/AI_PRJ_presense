import {
  buildDefaultPrimarySettings,
  buildDefaultSectionSettings,
} from "@/lib/app-data";
import { getSupabaseBrowserClient } from "@/lib/supabase/client";

export type PresentationRecord = {
  id: string;
  title: string;
  date: string;
  duration: string;
  result: string;
  stressAverage: string;
  diagnosis: string;
};

export type DiagnosisRecord = {
  id: string;
  label: string;
  value: string;
  detail: string;
};

export type PersistedSettings = {
  primary: Record<string, boolean>;
  sections: Record<string, boolean>;
};

type SavePracticeSessionInput = {
  durationSeconds: number;
  averageStress: number;
  result: string;
  diagnosis: string;
  nextAction: string;
  sceneLabel: string;
};

function formatDuration(durationSeconds: number) {
  const minutes = Math.floor(durationSeconds / 60);
  const seconds = durationSeconds % 60;
  return `${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
}

function formatSessionDate(timestamp: string) {
  return new Date(timestamp).toISOString().slice(0, 10);
}

function buildDefaultSettings(): PersistedSettings {
  return {
    primary: buildDefaultPrimarySettings(),
    sections: buildDefaultSectionSettings(),
  };
}

export async function ensureUserProfile(userId: string, email: string | undefined, fullName?: string) {
  const supabase = getSupabaseBrowserClient();

  if (!supabase) {
    return;
  }

  const { error } = await supabase.from("profiles").upsert(
    {
      id: userId,
      email: email ?? null,
      full_name: fullName ?? null,
    },
    { onConflict: "id" },
  );

  if (error) {
    throw error;
  }
}

export async function loadDashboardData() {
  const supabase = getSupabaseBrowserClient();

  if (!supabase) {
    return {
      presentationRecords: [],
      diagnosisHistory: [],
    };
  }

  const [{ data: recordRows, error: recordError }, { data: diagnosisRows, error: diagnosisError }] = await Promise.all([
    supabase
      .from("presentation_records")
      .select("id, title, session_date, duration_seconds, result, stress_average, diagnosis")
      .order("session_date", { ascending: false }),
    supabase
      .from("diagnosis_history")
      .select("id, label, value, detail, created_at")
      .order("created_at", { ascending: false }),
  ]);

  if (recordError) {
    throw recordError;
  }

  if (diagnosisError) {
    throw diagnosisError;
  }

  return {
    presentationRecords: (recordRows ?? []).map((row) => ({
      id: row.id,
      title: row.title,
      date: formatSessionDate(row.session_date),
      duration: formatDuration(row.duration_seconds),
      result: row.result,
      stressAverage: Number(row.stress_average).toFixed(2),
      diagnosis: row.diagnosis,
    })) as PresentationRecord[],
    diagnosisHistory: (diagnosisRows ?? []).map((row) => ({
      id: row.id,
      label: row.label,
      value: row.value,
      detail: row.detail,
    })) as DiagnosisRecord[],
  };
}

export async function loadUserSettings() {
  const supabase = getSupabaseBrowserClient();

  if (!supabase) {
    return buildDefaultSettings();
  }

  const { data, error } = await supabase.from("user_settings").select("primary_settings, section_settings").maybeSingle();

  if (error) {
    throw error;
  }

  if (!data) {
    const defaults = buildDefaultSettings();
    await saveUserSettings(defaults);
    return defaults;
  }

  return {
    primary: {
      ...buildDefaultPrimarySettings(),
      ...(data.primary_settings as Record<string, boolean> | null),
    },
    sections: {
      ...buildDefaultSectionSettings(),
      ...(data.section_settings as Record<string, boolean> | null),
    },
  };
}

export async function saveUserSettings(settings: PersistedSettings) {
  const supabase = getSupabaseBrowserClient();

  if (!supabase) {
    return;
  }

  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  if (userError) {
    throw userError;
  }

  if (!user) {
    throw new Error("No authenticated user found.");
  }

  const { error } = await supabase.from("user_settings").upsert(
    {
      user_id: user.id,
      primary_settings: settings.primary,
      section_settings: settings.sections,
    },
    { onConflict: "user_id" },
  );

  if (error) {
    throw error;
  }
}

export async function savePracticeSession(input: SavePracticeSessionInput) {
  const supabase = getSupabaseBrowserClient();

  if (!supabase) {
    return;
  }

  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  if (userError) {
    throw userError;
  }

  if (!user) {
    return;
  }

  const sessionDate = new Date().toISOString();
  const title = `${input.sceneLabel} Practice`;

  const { error: recordError } = await supabase.from("presentation_records").insert({
    user_id: user.id,
    title,
    session_date: sessionDate,
    duration_seconds: input.durationSeconds,
    result: input.result,
    stress_average: input.averageStress,
    diagnosis: input.diagnosis,
  });

  if (recordError) {
    throw recordError;
  }

  const diagnosisEntries = [
    {
      user_id: user.id,
      label: "Latest session result",
      value: input.result,
      detail: `${input.sceneLabel} rehearsal completed on ${formatSessionDate(sessionDate)}.`,
    },
    {
      user_id: user.id,
      label: "Average stress",
      value: input.averageStress.toFixed(2),
      detail: input.diagnosis,
    },
    {
      user_id: user.id,
      label: "Next coaching focus",
      value: input.nextAction,
      detail: "Generated from the most recent practice session.",
    },
  ];

  const { error: diagnosisError } = await supabase.from("diagnosis_history").insert(diagnosisEntries);

  if (diagnosisError) {
    throw diagnosisError;
  }
}
