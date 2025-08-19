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
import { SafeAreaView as DefaultSafeAreaView } from 'react-native-safe-area-context';

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
export type SafeAreaViewProps = ViewProps;
export type TextInputProps = ThemeProps & DefaultTextInput['props'];

export function useThemeColor(
  props: { light?: string; dark?: string },
  colorName: keyof typeof Colors.light & keyof typeof Colors.dark,
) {
  const theme = useColorScheme() ?? 'light';
  const colorFromProps = props[theme];

  if (colorFromProps) {
    return colorFromProps;
  } else {
    return Colors[theme][colorName];
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
      style={[style, type === 'title' ? { fontFamily: 'Poppins-Bold' } : {}]}
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
      style={[style, type === 'default' ? {} : {}]}
      className={`${className}`}
      {...otherProps}
    />
  );
}

export function ThemedSafeAreaView(props: SafeAreaViewProps) {
  const {
    style,
    lightColor,
    darkColor,
    className,
    type = 'default',
    ...otherProps
  } = props;

  const backgroundColor = useThemeColor(
    { light: lightColor, dark: darkColor },
    'background',
  );

  return (
    <DefaultSafeAreaView
      style={[style]}
      className={`flex-1 ${className} bg-background-500`}
      {...otherProps}
    />
  );
}

export function TextInput(props: TextInputProps) {
  const { style, lightColor, darkColor, className, ...otherProps } = props;
  const color = useThemeColor({ light: lightColor, dark: darkColor }, 'text');

  return (
    <DefaultTextInput
      style={[style]}
      placeholderTextColor="#5d5d5d"
      className={`${className} font-body`}
      {...otherProps}
    />
  );
}
