/**
 * Learn more about Light and Dark modes:
 * https://docs.expo.io/guides/color-schemes/
 */
import {
  Text as DefaultText,
  View as DefaultView,
  TextInput as DefaultTextInput,
  useColorScheme,
} from 'react-native';
import {
  SafeAreaView as DefaultSafeAreaView,
  SafeAreaViewProps,
} from 'react-native-safe-area-context';
import { Colors } from '@/constants/Colors';

type ThemeProps = {
  lightColor?: string;
  darkColor?: string;
};

export type TextProps = ThemeProps &
  DefaultText['props'] & {
    type?: 'default' | 'title';
  };

export type ViewProps = ThemeProps &
  DefaultView['props'] & {
    type?: 'default' | 'card';
  };

export type TextInputProps = ThemeProps & DefaultTextInput['props'];

// Define the available color keys from your Colors object
type ColorKeys = keyof typeof Colors.light | keyof typeof Colors.dark;

export function useThemeColor(
  props: { light?: string; dark?: string },
  colorName: ColorKeys,
) {
  const theme = useColorScheme() ?? 'light';
  const colorFromProps = props[theme];
  
  if (colorFromProps) {
    return colorFromProps;
  } else {
    // Use type assertion since we know the structure of our Colors object
    const themeColors = Colors[theme] as Record<string, any>;
    return themeColors[colorName as string];
  }
}

export function Text(props: TextProps) {
  const {
    style,
    lightColor,
    darkColor,
    className,
    type = 'default',
    ...otherProps
  } = props;
  
  const colorName = type === 'title' ? 'title' : 'text';
  const color = useThemeColor(
    { light: lightColor, dark: darkColor },
    colorName,
  );

  return (
    <DefaultText
      style={[{ color }, style, type === 'title' ? { fontFamily: 'Poppins-Bold' } : {}]}
      className={`${className} font-body`}
      {...otherProps}
    />
  );
}

export function View(props: ViewProps) {
  const {
    style,
    lightColor,
    darkColor,
    className,
    type = 'default',
    ...otherProps
  } = props;
  
  const colorName = type === 'card' ? 'cardBackground' : 'background';
  const backgroundColor = useThemeColor(
    { light: lightColor, dark: darkColor },
    colorName,
  );

  return (
    <DefaultView
      style={[{ backgroundColor }, style]}
      className={`${className}`}
      {...otherProps}
    />
  );
}

export function ThemedSafeAreaView(props: SafeAreaViewProps) {
  const { style, className, ...otherProps } = props;
  const backgroundColor = useThemeColor({}, 'background');
  
  return (
    <DefaultSafeAreaView
      style={[{ backgroundColor }, style]}
      className={`flex-1 ${className}`}
      {...otherProps}
    />
  );
}

export function TextInput(props: TextInputProps) {
  const { style, lightColor, darkColor, className, ...otherProps } = props;
  const color = useThemeColor({ light: lightColor, dark: darkColor }, 'text');
  const backgroundColor = useThemeColor({}, 'cardBackground');
  
  return (
    <DefaultTextInput
      style={[{ color, backgroundColor }, style]}
      placeholderTextColor="#5d5d5d"
      className={`${className} font-body`}
      {...otherProps}
    />
  );
}