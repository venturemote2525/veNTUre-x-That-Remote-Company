// components/AnimatedComponents.tsx
import React, { useEffect, useRef } from 'react';
import {
  Animated,
  Easing,
  Pressable,
  PressableProps,
  View,
  ViewProps,
  Text,
  TextProps,
  StyleProp,
  ViewStyle,
  ColorValue,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

// AnimatedPressable component
interface AnimatedPressableProps extends PressableProps {
  scaleAmount?: number;
  className?: string;
}

export const AnimatedPressable: React.FC<AnimatedPressableProps> = ({
  children,
  scaleAmount = 0.95,
  className = '',
  style,
  onPressIn,
  onPressOut,
  ...props
}) => {
  const scale = useRef(new Animated.Value(1)).current;

  const handlePressIn = (event: any) => {
    Animated.spring(scale, {
      toValue: scaleAmount,
      useNativeDriver: true,
      speed: 50,
    }).start();
    onPressIn?.(event);
  };

  const handlePressOut = (event: any) => {
    Animated.spring(scale, {
      toValue: 1,
      useNativeDriver: true,
      speed: 50,
    }).start();
    onPressOut?.(event);
  };

  return (
    <Animated.View style={{ transform: [{ scale }] }}>
      <Pressable
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        style={style}
        className={className}
        {...props}
      >
        {children}
      </Pressable>
    </Animated.View>
  );
};

// Fade in animation hook
export const useFadeIn = (duration: number = 300) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration,
      easing: Easing.out(Easing.ease),
      useNativeDriver: true,
    }).start();
  }, [fadeAnim, duration]);

  return {
    opacity: fadeAnim,
  };
};

// Slide in animation hook
export const useSlideIn = (duration: number = 300, from: 'top' | 'bottom' | 'left' | 'right' = 'bottom') => {
  const slideAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(slideAnim, {
      toValue: 1,
      duration,
      easing: Easing.out(Easing.ease),
      useNativeDriver: true,
    }).start();
  }, [slideAnim, duration]);

  const getTranslate = () => {
    switch (from) {
      case 'top':
        return [{ translateY: slideAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [-100, 0]
        }) }];
      case 'bottom':
        return [{ translateY: slideAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [100, 0]
        }) }];
      case 'left':
        return [{ translateX: slideAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [-100, 0]
        }) }];
      case 'right':
        return [{ translateX: slideAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [100, 0]
        }) }];
      default:
        return [{ translateY: slideAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [100, 0]
        }) }];
    }
  };

  return {
    transform: getTranslate(),
    opacity: slideAnim,
  };
};

// BouncingIcon component
interface BouncingIconProps {
  children: React.ReactNode;
  duration?: number;
}

export const BouncingIcon: React.FC<BouncingIconProps> = ({
  children,
  duration = 1000,
}) => {
  const bounceAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(bounceAnim, {
          toValue: -15,
          duration: duration / 2,
          easing: Easing.out(Easing.quad),
          useNativeDriver: true,
        }),
        Animated.timing(bounceAnim, {
          toValue: 0,
          duration: duration / 2,
          easing: Easing.in(Easing.quad),
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, [bounceAnim, duration]);

  return (
    <Animated.View style={{ transform: [{ translateY: bounceAnim }] }}>
      {children}
    </Animated.View>
  );
};

// GradientCard component
interface GradientCardProps extends ViewProps {
  colors?: readonly [ColorValue, ColorValue, ...ColorValue[]];
  start?: { x: number; y: number };
  end?: { x: number; y: number };
  className?: string;
}

export const GradientCard: React.FC<GradientCardProps> = ({
  children,
  colors = ['#667eea', '#764ba2'] as readonly [ColorValue, ColorValue],
  start = { x: 0, y: 0 },
  end = { x: 1, y: 1 },
  className = '',
  style,
  ...props
}) => {
  return (
    <LinearGradient
      colors={colors}
      start={start}
      end={end}
      className={`rounded-xl p-4 ${className}`}
      style={style}
      {...props}
    >
      {children}
    </LinearGradient>
  );
};

// AnimatedCounter component
interface AnimatedCounterProps extends TextProps {
  value: number;
  duration?: number;
}

export const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  value,
  duration = 1000,
  style,
  ...props
}) => {
  const animatedValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: value,
      duration,
      easing: Easing.out(Easing.exp),
      useNativeDriver: false,
    }).start();
  }, [value, duration, animatedValue]);

  const displayValue = animatedValue.interpolate({
    inputRange: [0, value],
    outputRange: ['0', Math.round(value).toString()],
  });

  return (
    <Animated.Text style={[style, { fontVariant: ['tabular-nums'] }]} {...props}>
      {displayValue}
    </Animated.Text>
  );
};

export default {
  AnimatedPressable,
  useFadeIn,
  useSlideIn,
  BouncingIcon,
  GradientCard,
  AnimatedCounter,
};