/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0914',
        surface: 'rgba(25, 22, 50, 0.4)',
        'surface-hover': 'rgba(255, 255, 255, 0.05)',
        border: 'rgba(255, 255, 255, 0.08)',
        primary: '#06b6d4',
        secondary: '#a0a0c0',
        danger: '#f43f5e',
        success: '#10b981',
        warning: '#f59e0b',
      },
      fontFamily: {
        sans: ['Outfit', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'glow-primary': 'linear-gradient(135deg, #06b6d4, #3b82f6)',
        'bg-gradient': 'radial-gradient(circle at 50% 50%, #171530 0%, #0a0914 100%)',
      }
    },
  },
  plugins: [],
}
