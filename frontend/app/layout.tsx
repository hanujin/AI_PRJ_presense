import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { SupabaseProvider } from "@/components/supabase-provider";
import { LanguageProvider } from "@/lib/i18n";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "PreSense Dashboard",
  description: "Monitoring dashboard for live stress and model-assisted presentation analysis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${jetbrainsMono.variable}`}>
        <LanguageProvider>
          <SupabaseProvider>{children}</SupabaseProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
