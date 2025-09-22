import { useColorScheme } from 'react-native';
import { Colors } from '@/constants/Colors';

/**
 * A hook that returns the appropriate color for the current theme.
 */
export function useThemeColor(
  props: { light?: string; dark?: string },
  colorName: keyof typeof Colors.light & keyof typeof Colors.dark
): string {
  const theme = useColorScheme() ?? 'light';
  const colorFromProps = props[theme];

  if (colorFromProps) {
    return colorFromProps;
  } else {
    // Get the color value from the theme
    const themeColors = Colors[theme];
    const colorValue = themeColors[colorName];
    
    // Return the color as string, handling the case where it might not exist
    return typeof colorValue === 'string' ? colorValue : Colors.light.background;
  }
}