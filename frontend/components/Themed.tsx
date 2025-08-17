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

import { Colors } from '@/constants/Colors';

type ThemeProps = {
  lightColor?: string;
  darkColor?: string;
};

export type TextProps = ThemeProps &
  DefaultText['props'] & {
    type?: 'default' | 'title' | 'defaultSemiBold' | 'subtitle' | 'link';
  };
export type ViewProps = ThemeProps & DefaultView['props'];
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
  const { style, lightColor, darkColor, className, ...otherProps } = props;
  const color = useThemeColor({ light: lightColor, dark: darkColor }, 'text');

  return (
    <DefaultText
      style={[{ color }, style]}
      className={`${className} font-body`}
      {...otherProps}
    />
  );
}

export function View(props: ViewProps) {
  // const { style, lightColor, darkColor, className, ...otherProps } = props;
  const { lightColor, darkColor, className, ...otherProps } = props;
  const backgroundColor = useThemeColor(
    { light: lightColor, dark: darkColor },
    'background',
  );

  return (
    <DefaultView
      // style={[{ backgroundColor }, style]}
      className={`${className} bg-${backgroundColor}`}
      {...otherProps}
    />
  );
}

export function TextInput(props: TextInputProps) {
  // const { style, lightColor, darkColor, className, ...otherProps } = props;
  const { lightColor, darkColor, className, ...otherProps } = props;
  const color = useThemeColor({ light: lightColor, dark: darkColor }, 'text');

  return (
    <DefaultTextInput
      // style={[{ color }, style]}
      placeholderTextColor="#5d5d5d"
      className={`${className} font-body text-${color}`}
      {...otherProps}
    />
  );
}
