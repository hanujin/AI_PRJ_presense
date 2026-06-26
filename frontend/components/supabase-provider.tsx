"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { Session, User } from "@supabase/supabase-js";
import { getSupabaseBrowserClient, hasSupabaseEnv } from "@/lib/supabase/client";
import { ensureUserProfile } from "@/lib/supabase/data";

type AuthContextValue = {
  hasEnv: boolean;
  isLoading: boolean;
  user: User | null;
  session: Session | null;
  errorMessage: string;
  signInWithPassword: (email: string, password: string) => Promise<void>;
  signUpWithPassword: (email: string, password: string, fullName: string) => Promise<{ needsEmailConfirmation: boolean }>;
  signOut: () => Promise<void>;
};

const SupabaseAuthContext = createContext<AuthContextValue | null>(null);

export function SupabaseProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(hasSupabaseEnv);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const supabase = getSupabaseBrowserClient();

    if (!supabase) {
      setIsLoading(false);
      return;
    }

    let isMounted = true;

    void supabase.auth.getSession().then(({ data, error }) => {
      if (!isMounted) {
        return;
      }

      if (error) {
        setErrorMessage(error.message);
      }

      setSession(data.session ?? null);
      setUser(data.session?.user ?? null);
      setIsLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession ?? null);
      setUser(nextSession?.user ?? null);
      setIsLoading(false);
    });

    return () => {
      isMounted = false;
      subscription.unsubscribe();
    };
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      hasEnv: hasSupabaseEnv,
      isLoading,
      user,
      session,
      errorMessage,
      async signInWithPassword(email, password) {
        const supabase = getSupabaseBrowserClient();

        if (!supabase) {
          throw new Error("Supabase environment variables are missing.");
        }

        const { error } = await supabase.auth.signInWithPassword({ email, password });

        if (error) {
          throw error;
        }
      },
      async signUpWithPassword(email, password, fullName) {
        const supabase = getSupabaseBrowserClient();

        if (!supabase) {
          throw new Error("Supabase environment variables are missing.");
        }

        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: {
              full_name: fullName,
            },
          },
        });

        if (error) {
          throw error;
        }

        if (data.user) {
          await ensureUserProfile(data.user.id, data.user.email, fullName);
        }

        return {
          needsEmailConfirmation: !data.session,
        };
      },
      async signOut() {
        const supabase = getSupabaseBrowserClient();

        if (!supabase) {
          return;
        }

        const { error } = await supabase.auth.signOut();

        if (error) {
          throw error;
        }
      },
    }),
    [errorMessage, isLoading, session, user],
  );

  return <SupabaseAuthContext.Provider value={value}>{children}</SupabaseAuthContext.Provider>;
}

export function useSupabaseAuth() {
  const context = useContext(SupabaseAuthContext);

  if (!context) {
    throw new Error("useSupabaseAuth must be used within SupabaseProvider.");
  }

  return context;
}
