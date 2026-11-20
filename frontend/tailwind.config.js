/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        lab: {
          page: '#070d17',
          panel: '#0d1726',
          raised: '#13213a',
          stage: '#04080f',
          line: 'rgb(148 178 214 / 0.14)',
          strong: 'rgb(148 178 214 / 0.28)',
        },
        ink: {
          1: '#e8eef6',
          2: '#a9b7c9',
          3: '#8193ab',
        },
        accent: {
          DEFAULT: '#2dd4bf',
          soft: '#5eead4',
          deep: '#0f766e',
          ink: '#04121a',
        },
        // Categorical model identity, validated for CVD separation on lab.panel.
        series: {
          1: '#3987e5',
          2: '#d95926',
          3: '#199e70',
        },
        caution: '#fbbf24',
        danger: '#f87171',
      },
      fontFamily: {
        sans: ['"Inter Variable"', 'system-ui', '-apple-system', '"Segoe UI"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        panel: '0 1px 0 0 rgb(255 255 255 / 0.04) inset, 0 12px 32px -16px rgb(0 0 0 / 0.6)',
        glow: '0 0 0 1px rgb(45 212 191 / 0.35), 0 8px 28px -12px rgb(45 212 191 / 0.45)',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          from: { backgroundPosition: '-200% 0' },
          to: { backgroundPosition: '200% 0' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        'grow-x': {
          from: { transform: 'scaleX(0)' },
          to: { transform: 'scaleX(1)' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
      },
      animation: {
        'fade-up': 'fade-up 420ms cubic-bezier(0.2, 0.7, 0.2, 1) both',
        shimmer: 'shimmer 1.6s linear infinite',
        scan: 'scan 1.8s cubic-bezier(0.4, 0, 0.2, 1) infinite',
        'grow-x': 'grow-x 900ms cubic-bezier(0.2, 0.7, 0.2, 1) both',
        'fade-in': 'fade-in 500ms ease-out both',
      },
    },
  },
  plugins: [],
};
