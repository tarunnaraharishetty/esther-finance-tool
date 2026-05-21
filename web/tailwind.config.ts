import type { Config } from "tailwindcss";
import animate from "tailwindcss-animate";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "1rem",
      screens: { "2xl": "1440px" },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        bull: "hsl(var(--bull))",
        bear: "hsl(var(--bear))",
        warn: "hsl(var(--warn))",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
        xl: "calc(var(--radius) + 4px)",
        "2xl": "calc(var(--radius) + 8px)",
      },
      fontFamily: {
        sans: ['"Inter"', "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ['"JetBrains Mono"', '"SF Mono"', "Consolas", "monospace"],
        display: ['"Inter"', "sans-serif"],
      },
      letterSpacing: {
        tightest: "-0.04em",
      },
      backgroundImage: {
        "gradient-bull":
          "linear-gradient(135deg, hsl(var(--bull) / 0.18), hsl(var(--bull) / 0.02))",
        "gradient-bear":
          "linear-gradient(135deg, hsl(var(--bear) / 0.18), hsl(var(--bear) / 0.02))",
        "gradient-primary":
          "linear-gradient(135deg, hsl(var(--primary) / 0.2), hsl(var(--primary) / 0.02))",
        "gradient-panel":
          "linear-gradient(180deg, hsl(var(--card)) 0%, hsl(var(--background)) 100%)",
        "gradient-mesh":
          "radial-gradient(at 0% 0%, hsl(var(--primary) / 0.18) 0px, transparent 50%), radial-gradient(at 100% 0%, hsl(var(--accent) / 0.18) 0px, transparent 50%), radial-gradient(at 50% 100%, hsl(var(--bull) / 0.08) 0px, transparent 50%)",
      },
      boxShadow: {
        glow: "0 0 0 1px hsl(var(--primary) / 0.15), 0 12px 32px -8px hsl(var(--primary) / 0.35)",
        "glow-bull":
          "0 0 0 1px hsl(var(--bull) / 0.18), 0 14px 30px -10px hsl(var(--bull) / 0.3)",
        "glow-bear":
          "0 0 0 1px hsl(var(--bear) / 0.18), 0 14px 30px -10px hsl(var(--bear) / 0.3)",
        inset: "inset 0 1px 0 0 hsl(var(--foreground) / 0.06)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in-fast": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "slide-up": {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        // Aurora-like background drift for hero cards. Slow + subtle.
        aurora: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        "ping-soft": {
          "0%": { transform: "scale(1)", opacity: "0.6" },
          "100%": { transform: "scale(2)", opacity: "0" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.3s ease-out",
        "fade-in-fast": "fade-in-fast 0.18s ease-out",
        "slide-up": "slide-up 0.35s cubic-bezier(0.16, 1, 0.3, 1)",
        shimmer: "shimmer 2.2s linear infinite",
        aurora: "aurora 18s ease infinite",
        "ping-soft": "ping-soft 2s cubic-bezier(0, 0, 0.2, 1) infinite",
      },
    },
  },
  plugins: [animate],
} satisfies Config;
