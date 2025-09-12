
export const Colors = {
  light: {
    // Keep your existing colors for compatibility
    text: '#11181C',
    title: '#0077b6', 
    background: '#F2F2F2',
    cardBackground: '#FFFFFF',
    secondary: '#0077b6',
    primary: '#000000',
    
    // Enhanced color palette
    colors: {
      primary: {
        50: '#f0f9ff',
        100: '#e0f2fe',
        200: '#bae6fd',
        300: '#7dd3fc',
        400: '#38bdf8',
        500: '#0ea5e9',
        600: '#0284c7',
        700: '#0369a1',
        800: '#075985',
        900: '#0c4a6e',
      },
      secondary: {
        50: '#f8fafc',
        100: '#f1f5f9',
        200: '#e2e8f0',
        300: '#cbd5e1',
        400: '#94a3b8',
        500: '#64748b',
        600: '#475569',
        700: '#334155',
        800: '#1e293b',
        900: '#0f172a',
      },
      success: {
        50: '#f0fdf4',
        100: '#dcfce7',
        200: '#bbf7d0',
        500: '#22c55e',
        600: '#16a34a',
        700: '#15803d',
      },
      warning: {
        50: '#fffbeb',
        100: '#fef3c7',
        200: '#fde68a',
        500: '#f59e0b',
        600: '#d97706',
        700: '#b45309',
      },
      error: {
        50: '#fef2f2',
        100: '#fee2e2',
        200: '#fecaca',
        500: '#ef4444',
        600: '#dc2626',
        700: '#b91c1c',
      },
    },
    
    // Gradient combinations (TypeScript-safe with 'as const')
    gradients: {
      primary: ['#667eea', '#764ba2'] as const,
      secondary: ['#0077b6', '#00b4d8'] as const,
      success: ['#11998e', '#38ef7d'] as const,
      sunset: ['#ff9a9e', '#fecfef'] as const,
      ocean: ['#667eea', '#764ba2'] as const,
      fire: ['#ff6b6b', '#feca57'] as const,
      purple: ['#a8edea', '#fed6e3'] as const,
      blue: ['#74b9ff', '#0984e3'] as const,
      health: ['#00cec9', '#55a3ff'] as const,
      food: ['#fd79a8', '#fdcb6e'] as const,
    },
  },
  dark: {
    // Keep your existing colors for compatibility
    text: '#FFFFFF',
    title: '#0096c7',
    background: '#1a1a1a',
    cardBackground: '#2a2a2a',
    secondary: '#0096c7',
    primary: '#FFFFFF',
    
    // Enhanced color palette (dark mode variants)
    colors: {
      primary: {
        50: '#0c4a6e',
        100: '#075985',
        200: '#0369a1',
        300: '#0284c7',
        400: '#0ea5e9',
        500: '#38bdf8',
        600: '#7dd3fc',
        700: '#bae6fd',
        800: '#e0f2fe',
        900: '#f0f9ff',
      },
      secondary: {
        50: '#0f172a',
        100: '#1e293b',
        200: '#334155',
        300: '#475569',
        400: '#64748b',
        500: '#94a3b8',
        600: '#cbd5e1',
        700: '#e2e8f0',
        800: '#f1f5f9',
        900: '#f8fafc',
      },
      success: {
        50: '#15803d',
        100: '#16a34a',
        200: '#22c55e',
        500: '#22c55e',
        600: '#bbf7d0',
        700: '#dcfce7',
      },
      warning: {
        50: '#b45309',
        100: '#d97706',
        200: '#f59e0b',
        500: '#f59e0b',
        600: '#fde68a',
        700: '#fef3c7',
      },
      error: {
        50: '#b91c1c',
        100: '#dc2626',
        200: '#ef4444',
        500: '#ef4444',
        600: '#fecaca',
        700: '#fee2e2',
      },
    },
    
    // Dark mode gradients (TypeScript-safe with 'as const')
    gradients: {
      primary: ['#4c1d95', '#7c3aed'] as const,
      secondary: ['#1e293b', '#475569'] as const,
      success: ['#065f46', '#047857'] as const,
      sunset: ['#be185d', '#ec4899'] as const,
      ocean: ['#1e293b', '#0f172a'] as const,
      fire: ['#dc2626', '#b91c1c'] as const,
      purple: ['#581c87', '#7c2d12'] as const,
      blue: ['#1e40af', '#1e3a8a'] as const,
      health: ['#047857', '#0d9488'] as const,
      food: ['#be185d', '#a21caf'] as const,
    },
  },
} as const;

// Typography scale
export const Typography = {
  fontSizes: {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 18,
    xl: 20,
    '2xl': 24,
    '3xl': 30,
    '4xl': 36,
    '5xl': 48,
  },
  fontWeights: {
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
    extrabold: '800',
  },
  lineHeights: {
    tight: 1.25,
    normal: 1.5,
    relaxed: 1.75,
  },
} as const;

// Shadow presets
export const Shadows = {
  small: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  medium: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 5,
  },
  large: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 16,
    elevation: 10,
  },
} as const;

// Animation presets
export const Animations = {
  spring: {
    damping: 15,
    stiffness: 150,
    mass: 1,
  },
  timing: {
    duration: 300,
  },
  bounce: {
    damping: 8,
    stiffness: 100,
    mass: 1,
  },
} as const;