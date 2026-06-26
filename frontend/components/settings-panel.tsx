"use client";

import { useEffect, useState } from "react";
import { Bell, Camera, Languages, Lock, Mic, SlidersHorizontal } from "lucide-react";
import {
  buildDefaultPrimarySettings,
  buildDefaultSectionSettings,
  primarySettings,
  settingsSections,
} from "@/lib/app-data";
import { useSupabaseAuth } from "@/components/supabase-provider";
import { useLang } from "@/lib/i18n";
import { loadUserSettings, saveUserSettings } from "@/lib/supabase/data";

const PRIMARY_SETTINGS_KEY = "presense-primary-settings";
const SECTION_SETTINGS_KEY = "presense-section-settings";

const PRIMARY_ICONS = [Camera, Mic, Bell];

export function SettingsPanel() {
  const { hasEnv, user } = useSupabaseAuth();
  const { lang, setLang, t } = useLang();
  const [primaryToggles, setPrimaryToggles] = useState<Record<string, boolean>>(() => buildDefaultPrimarySettings());
  const [sectionToggles, setSectionToggles] = useState<Record<string, boolean>>(() => buildDefaultSectionSettings());
  const [statusMessage, setStatusMessage] = useState("");
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    if (!hasEnv || !user) { setLoadError(""); return; }
    let isMounted = true;

    void loadUserSettings()
      .then((settings) => {
        if (!isMounted) return;
        setPrimaryToggles(settings.primary);
        setSectionToggles(settings.sections);
        setLoadError("");
      })
      .catch((error) => {
        if (!isMounted) return;
        setLoadError(error instanceof Error ? error.message : t("settings.failedLoad"));
      });

    return () => { isMounted = false; };
  }, [hasEnv, user]);

  useEffect(() => {
    if (hasEnv && user) return;
    const p = window.localStorage.getItem(PRIMARY_SETTINGS_KEY);
    const s = window.localStorage.getItem(SECTION_SETTINGS_KEY);
    if (p) setPrimaryToggles((c) => ({ ...c, ...JSON.parse(p) }));
    if (s) setSectionToggles((c) => ({ ...c, ...JSON.parse(s) }));
  }, [hasEnv, user]);

  useEffect(() => {
    if (hasEnv && user) return;
    window.localStorage.setItem(PRIMARY_SETTINGS_KEY, JSON.stringify(primaryToggles));
  }, [hasEnv, primaryToggles, user]);

  useEffect(() => {
    if (hasEnv && user) return;
    window.localStorage.setItem(SECTION_SETTINGS_KEY, JSON.stringify(sectionToggles));
  }, [hasEnv, sectionToggles, user]);

  async function persistSettings(nextP: Record<string, boolean>, nextS: Record<string, boolean>) {
    if (!hasEnv || !user) { setStatusMessage(t("settings.savedBrowserOnly")); return; }
    try {
      await saveUserSettings({ primary: nextP, sections: nextS });
      setStatusMessage(t("settings.savedAccount"));
      setLoadError("");
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : t("settings.failedSave"));
    }
  }

  function togglePrimary(id: string) {
    const next = { ...primaryToggles, [id]: !primaryToggles[id] };
    setPrimaryToggles(next);
    void persistSettings(next, sectionToggles);
  }

  function toggleSection(id: string) {
    const next = { ...sectionToggles, [id]: !sectionToggles[id] };
    setSectionToggles(next);
    void persistSettings(primaryToggles, next);
  }

  const SECTION_ICONS = [SlidersHorizontal, Lock, Bell];

  return (
    <main className="page-content">
      <div className="page-top">
        <div>
          <h1 className="page-title">{t("settings.title")}</h1>
          <p className="page-subtitle">{t("settings.subtitle")}</p>
        </div>
      </div>

      {loadError && <p className="error-msg" style={{ marginBottom: 16 }}>{loadError}</p>}
      {statusMessage && <p className="success-msg" style={{ marginBottom: 16 }}>{statusMessage}</p>}

      {/* Language */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="settings-row">
          <div>
            <div className="settings-row-label" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Languages size={15} color="var(--text-secondary)" />
              {t("settings.language")}
            </div>
            <div className="kpi-sub" style={{ marginTop: 2 }}>{t("settings.languageDesc")}</div>
          </div>
          <div className="lang-segment">
            <button
              type="button"
              className={lang === "ko" ? "active" : ""}
              onClick={() => setLang("ko")}
            >
              한국어
            </button>
            <button
              type="button"
              className={lang === "en" ? "active" : ""}
              onClick={() => setLang("en")}
            >
              English
            </button>
          </div>
        </div>
      </div>

      {/* Primary toggles */}
      <div className="kpi-row" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
        {primarySettings.map((setting, i) => {
          const Icon = PRIMARY_ICONS[i] ?? Camera;
          const isOn = primaryToggles[setting.id];
          return (
            <div key={setting.id} className="kpi-card">
              <div className="kpi-top">
                <div className="kpi-icon" style={isOn ? { background: "var(--accent-ok-light)", color: "var(--accent-ok)" } : {}}>
                  <Icon size={16} />
                </div>
                <button
                  type="button"
                  className={`toggle-btn${isOn ? " on" : ""}`}
                  aria-pressed={isOn}
                  onClick={() => togglePrimary(setting.id)}
                >
                  {isOn ? t("toggle.enabled") : t("toggle.disabled")}
                </button>
              </div>
              <div className="kpi-value" style={{ fontSize: 15, fontFamily: "inherit", marginBottom: 4 }}>{t(`set.${setting.id}`, setting.label)}</div>
              <div className="kpi-sub">{t("settings.clickToToggle")}</div>
            </div>
          );
        })}
      </div>

      {/* Section settings */}
      <div className="settings-grid">
        {settingsSections.map((section, i) => {
          const Icon = SECTION_ICONS[i] ?? SlidersHorizontal;
          return (
            <div key={section.title} className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">{t(`set.section.${section.title}`, section.title)}</span>
                </div>
                <Icon size={16} color="var(--text-secondary)" />
              </div>

              {section.items.map((item) => {
                const isOn = sectionToggles[item.id];
                return (
                  <div key={item.id} className="settings-row">
                    <div>
                      <div className="settings-row-label">{t(`set.${item.id}`, item.label)}</div>
                    </div>
                    <button
                      type="button"
                      className={`toggle-btn${isOn ? " on" : ""}`}
                      aria-pressed={isOn}
                      onClick={() => toggleSection(item.id)}
                    >
                      {isOn ? t("toggle.on") : t("toggle.off")}
                    </button>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </main>
  );
}
