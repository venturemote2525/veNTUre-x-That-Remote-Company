// components/AnimatedComponents.tsx
import React, { useEffect, useState } from 'react';
import { Pressable, ViewStyle, TextStyle } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
  runOnJS,
  withSequence,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { Colors, Shadows, Animations } from '@/constants/Colors';
import { Text } from '@/components/Themed';

// Animated Pressable with scale and haptic feedback
interface AnimatedPressableProps {
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
  className?: string;
  disabled?: boolean;
  haptic?: boolean;
  scaleAmount?: number;
}

export const AnimatedPressable: React.FC<AnimatedPressableProps> = ({
  children,
  onPress,
  style,
  className,
  disabled = false,
  haptic = true,
  scaleAmount = 0.97,
}) => {
  const scale = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const handlePressIn = () => {
    scale.value = withSpring(scaleAmount, Animations.spring);
    if (haptic) {
      runOnJS(Haptics.impactAsync)(Haptics.ImpactFeedbackStyle.Light);
    }
  };

  const handlePressOut = () => {
    scale.value = withSpring(1, Animations.spring);
  };

  const handlePress = () => {
    if (haptic) {
      runOnJS(Haptics.impactAsync)(Haptics.ImpactFeedbackStyle.Medium);
    }
    onPress?.();
  };

  return (
    <Pressable
      onPress={handlePress}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      disabled={disabled}
      style={style}
      className={className}
    >
      <Animated.View style={animatedStyle}>
        {children}
      </Animated.View>
    </Pressable>
  );
};

// Gradient Card with glass morphism effect
interface GradientCardProps {
  children: React.ReactNode;
  gradient?: readonly [string, string, ...string[]];
  style?: ViewStyle;
  glassEffect?: boolean;
  shadow?: keyof typeof Shadows;
}

export const GradientCard: React.FC<GradientCardProps> = ({
  children,
  gradient = Colors.light.gradients.primary,
  style,
  glassEffect = false,
  shadow = 'medium',
}) => {
  if (glassEffect) {
    return (
      <Animated.View
        style={[
          {
            borderRadius: 16,
            overflow: 'hidden',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            borderColor: 'rgba(255, 255, 255, 0.2)',
          },
          Shadows[shadow],
          style,
        ]}
      >
        {children}
      </Animated.View>
    );
  }

  return (
    <Animated.View style={[{ borderRadius: 16, overflow: 'hidden' }, Shadows[shadow], style]}>
      <LinearGradient colors={gradient} style={{ flex: 1 }}>
        {children}
      </LinearGradient>
    </Animated.View>
  );
};

// Animated Counter
interface AnimatedCounterProps {
  value: number;
  style?: TextStyle;
  duration?: number;
}

export const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  value,
  style,
  duration = 1000,
}) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let startValue = displayValue;
    let startTime: number | null = null;
    let animationFrame: number;

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      
      const currentValue = Math.round(startValue + (value - startValue) * progress);
      setDisplayValue(currentValue);

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [value, duration]);

  return <Text style={style}>{displayValue}</Text>;
};

// Fade In Animation Hook
export const useFadeIn = (delay: number = 0) => {
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(20);

  useEffect(() => {
    const timeout = setTimeout(() => {
      opacity.value = withTiming(1, { duration: 600 });
      translateY.value = withTiming(0, { duration: 600 });
    }, delay);

    return () => clearTimeout(timeout);
  }, [delay]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: translateY.value }],
  }));

  return animatedStyle;
};

// Slide In From Direction
export const useSlideIn = (direction: 'left' | 'right' | 'up' | 'down', delay: number = 0) => {
  const translateX = useSharedValue(direction === 'left' ? -100 : direction === 'right' ? 100 : 0);
  const translateY = useSharedValue(direction === 'up' ? -100 : direction === 'down' ? 100 : 0);
  const opacity = useSharedValue(0);

  useEffect(() => {
    const timeout = setTimeout(() => {
      translateX.value = withSpring(0, Animations.spring);
      translateY.value = withSpring(0, Animations.spring);
      opacity.value = withTiming(1, { duration: 400 });
    }, delay);

    return () => clearTimeout(timeout);
  }, [delay, direction]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
    ],
  }));

  return animatedStyle;
};

// Loading Pulse Animation
export const LoadingPulse: React.FC<{ style?: ViewStyle }> = ({ style }) => {
  const opacity = useSharedValue(0.3);

  useEffect(() => {
    const animate = () => {
      opacity.value = withSequence(
        withTiming(1, { duration: 800 }),
        withTiming(0.3, { duration: 800 })
      );
    };

    animate();
    const interval = setInterval(animate, 1600);
    return () => clearInterval(interval);
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
  }));

  return (
    <Animated.View
      style={[
        {
          backgroundColor: '#e2e8f0',
          borderRadius: 8,
        },
        animatedStyle,
        style,
      ]}
    />
  );
};

// Bouncing Icon Animation
interface BouncingIconProps {
  children: React.ReactNode;
  delay?: number;
}

export const BouncingIcon: React.FC<BouncingIconProps> = ({ children, delay = 0 }) => {
  const scale = useSharedValue(1);

  useEffect(() => {
    const startAnimation = () => {
      scale.value = withSequence(
        withSpring(1.2, Animations.bounce),
        withSpring(1, Animations.bounce)
      );
    };

    const timeout = setTimeout(startAnimation, delay);
    const interval = setInterval(startAnimation, 3000);

    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  }, [delay]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  return <Animated.View style={animatedStyle}>{children}</Animated.View>;
};

// Shake Animation for errors
export const useShakeAnimation = () => {
  const translateX = useSharedValue(0);

  const shake = () => {
    translateX.value = withSequence(
      withTiming(-10, { duration: 50 }),
      withTiming(10, { duration: 50 }),
      withTiming(-10, { duration: 50 }),
      withTiming(10, { duration: 50 }),
      withTiming(0, { duration: 50 })
    );
  };

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: translateX.value }],
  }));

  return { animatedStyle, shake };
};