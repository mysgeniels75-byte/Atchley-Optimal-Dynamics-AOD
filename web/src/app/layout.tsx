import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Triadic Consciousness Engine - AOD Theory",
  description: "Interactive visualization of the Triadic Consciousness Engine demonstrating Atchley Optimal Dynamics theory through cognitive phase dynamics",
  authors: [{ name: "Devin Earl Atchley" }],
  keywords: ["consciousness", "AI", "neural networks", "complex systems", "AOD theory", "cognitive science"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
