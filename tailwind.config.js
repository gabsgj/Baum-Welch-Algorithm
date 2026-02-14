/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './hmm_service/templates/**/*.html',
    './hmm_service/static/**/*.js'
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace']
      },
      colors: {
        palette: {
          white: '#F7F6F2',
          warm: '#F0E5CF',
          gray: '#C8C6C6',
          blue: '#4B6587'
        },
        sapphire: {
          50: '#eff4ff',
          100: '#dbe6fe',
          200: '#bfd3fe',
          300: '#93b4fd',
          400: '#6090fa',
          500: '#2563eb',
          600: '#1d4ed8',
          700: '#1e40af',
          800: '#1e3a8a',
          900: '#1a2f6b'
        },
        academic: {
          bg: '#F7F6F2',
          card: '#FFFFFF',
          border: '#C8C6C6',
          muted: '#999999'
        }
      }
    }
  },
  plugins: []
};
