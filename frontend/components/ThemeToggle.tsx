import React, { useEffect, useState } from 'react';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSpring,
} from 'react-native-reanimated';
import { View, Text } from '@/components/Themed';
import { useThemeMode } from '@/context/ThemeContext';
import { Colors, Shadows, Animations } from '@/constants/Colors';
import { AnimatedPressable } from '@/components/AnimatedComponents';

export default function ThemeToggle() {
  const { mode, setMode } = useThemeMode();
  const [containerWidth, setContainerWidth] = useState(0);
  const position = useSharedValue(0); // 0 = light, 1 = dark
  const scale = useSharedValue(1);

  useEffect(() => {
    position.value = withSpring(mode === 'light' ? 0 : 1, Animations.spring);
  }, [mode]);

  const sliderStyle = useAnimatedStyle(() => ({
    transform: [
      {
        translateX: withSpring(
          (containerWidth / 2 - 8) * position.value,
          Animations.spring,
        ),
      },
      { scale: scale.value },
    ],
  }));

  const handlePress = (newMode: 'light' | 'dark') => {
    // Create a proper spring config without duration (spring animations don't use duration)
    const springConfig = {
      damping: Animations.spring.damping,
      stiffness: Animations.spring.stiffness,
      mass: Animations.spring.mass,
    };

    scale.value = withSpring(0.9, springConfig, () => {
      scale.value = withSpring(1, springConfig);
    });
    setMode(newMode);
  };

  return (
    <View
      style={{
        position: 'relative',
        flexDirection: 'row',
        borderRadius: 16,
        padding: 4,
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        ...Shadows.small,
      }}
      onLayout={e => setContainerWidth(e.nativeEvent.layout.width)}>
      {/* Sliding pill with gradient */}
      {containerWidth > 0 && (
        <Animated.View
          style={[
            sliderStyle,
            {
              position: 'absolute',
              top: 4,
              bottom: 4,
              left: 4,
              width: containerWidth / 2 - 8,
              borderRadius: 12,
              overflow: 'hidden',
            },
          ]}>
          <LinearGradient
            colors={
              mode === 'light'
                ? Colors.light.gradients.sunset
                : Colors.light.gradients.ocean
            }
            style={{ flex: 1, borderRadius: 12 }}
          />
        </Animated.View>
      )}

      <AnimatedPressable
        onPress={() => handlePress('light')}
        style={{
          flex: 1,
          alignItems: 'center',
          justifyContent: 'center',
          paddingVertical: 12,
        }}>
        <Text
          style={{
            fontWeight: mode === 'light' ? '700' : '500',
            color:
              mode === 'light' ? 'white' : Colors.light.colors.secondary[400],
            fontSize: 14,
            textShadowColor:
              mode === 'light' ? 'rgba(0, 0, 0, 0.3)' : 'transparent',
            textShadowOffset: { width: 1, height: 1 },
            textShadowRadius: 2,
          }}>
          ☀️ Light
        </Text>
      </AnimatedPressable>

      <AnimatedPressable
        onPress={() => handlePress('dark')}
        style={{
          flex: 1,
          alignItems: 'center',
          justifyContent: 'center',
          paddingVertical: 12,
        }}>
        <Text
          style={{
            fontWeight: mode === 'dark' ? '700' : '500',
            color:
              mode === 'dark' ? 'white' : Colors.light.colors.secondary[400],
            fontSize: 14,
            textShadowColor:
              mode === 'dark' ? 'rgba(0, 0, 0, 0.3)' : 'transparent',
            textShadowOffset: { width: 1, height: 1 },
            textShadowRadius: 2,
          }}>
          🌙 Dark
        </Text>
      </AnimatedPressable>
    </View>
  );
}
