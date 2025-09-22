// components/ThemeToggle.tsx
import React, { useState, useEffect } from 'react';
import { LinearGradient } from 'expo-linear-gradient';
import { View, Text, StyleSheet } from 'react-native';
import { useThemeMode } from '@/context/ThemeContext';
import { Colors, Shadows } from '@/constants/Colors';
import { AnimatedPressable, useSlideIn } from '@/components/AnimatedComponents';

export default function ThemeToggle() {
  const { mode, setMode } = useThemeMode();
  const [containerWidth, setContainerWidth] = useState(0);
  const [position, setPosition] = useState(mode === 'light' ? 0 : 1);

  useEffect(() => {
    setPosition(mode === 'light' ? 0 : 1);
  }, [mode]);

  const handlePress = (newMode: 'light' | 'dark') => {
    setMode(newMode);
    setPosition(newMode === 'light' ? 0 : 1);
  };

  return (
    <View
      style={styles.container}
      onLayout={e => setContainerWidth(e.nativeEvent.layout.width)}
    >
      {/* Sliding pill */}
      {containerWidth > 0 && (
        <AnimatedPressable
          style={[
            styles.slider,
            {
              width: containerWidth / 2 - 8,
              transform: [{ translateX: (containerWidth / 2) * position }],
            },
          ]}
        >
          <LinearGradient
            colors={
              mode === 'light'
                ? Colors.light.gradients.sunset
                : Colors.light.gradients.ocean
            }
            style={styles.gradient}
          />
        </AnimatedPressable>
      )}

      {/* Light button */}
      <AnimatedPressable
        onPress={() => handlePress('light')}
        scaleAmount={0.9}
        style={styles.button}
      >
        <Text
          style={[
            styles.text,
            {
              fontWeight: mode === 'light' ? '700' : '500',
              color: mode === 'light' ? 'white' : Colors.light.colors.secondary[400],
              textShadowColor: mode === 'light' ? 'rgba(0,0,0,0.3)' : 'transparent',
            },
          ]}
        >
          ☀️ Light
        </Text>
      </AnimatedPressable>

      {/* Dark button */}
      <AnimatedPressable
        onPress={() => handlePress('dark')}
        scaleAmount={0.9}
        style={styles.button}
      >
        <Text
          style={[
            styles.text,
            {
              fontWeight: mode === 'dark' ? '700' : '500',
              color: mode === 'dark' ? 'white' : Colors.light.colors.secondary[400],
              textShadowColor: mode === 'dark' ? 'rgba(0,0,0,0.3)' : 'transparent',
            },
          ]}
        >
          🌙 Dark
        </Text>
      </AnimatedPressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    flexDirection: 'row',
    borderRadius: 16,
    padding: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    ...Shadows.small,
  },
  slider: {
    position: 'absolute',
    top: 4,
    bottom: 4,
    left: 4,
    borderRadius: 12,
    overflow: 'hidden',
  },
  gradient: {
    flex: 1,
    borderRadius: 12,
  },
  button: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
  },
  text: {
    fontSize: 14,
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
});
