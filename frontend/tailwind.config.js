/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0a0a1a',
        'bg-secondary': '#1a1a3e',
        'bg-card': 'rgba(26, 26, 62, 0.6)',
        'neon-blue': '#00d4ff',
        'neon-blue-dark': '#0099ff',
        'neon-purple': '#ff00ff',
        'neon-purple-dark': '#9d00ff',
        'neon-green': '#00ff88',
        'neon-yellow': '#ffff00',
        'neon-red': '#ff3366',
      },
      boxShadow: {
        'neon-blue': '0 0 20px rgba(0, 212, 255, 0.6)',
        'neon-purple': '0 0 20px rgba(255, 0, 255, 0.6)',
        'neon-green': '0 0 20px rgba(0, 255, 136, 0.6)',
        'glass': '0 0 30px rgba(0, 212, 255, 0.1)',
      },
      backdropBlur: {
        'glass': '20px',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(0, 212, 255, 0.4)' },
          '100%': { boxShadow: '0 0 40px rgba(0, 212, 255, 0.8)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  plugins: [],
}
