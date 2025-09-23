import { View, Text } from '@/components/Themed';
import { Pressable } from 'react-native';
import { useThemeMode } from '@/context/ThemeContext';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
} from 'react-native-reanimated';
import { useEffect, useState } from 'react';

export default function ThemeToggle() {
  const { mode, setMode } = useThemeMode();
  const [containerWidth, setContainerWidth] = useState(0);
  const position = useSharedValue(0); // 0 = light, 1 = dark

  useEffect(() => {
    position.value = withTiming(mode === 'light' ? 0 : 1, { duration: 100 });
  }, [mode]);

  const sliderStyle = useAnimatedStyle(() => ({
    transform: [
      {
        translateX: withTiming((containerWidth / 2) * position.value, {
          duration: 200,
        }),
      },
    ],
  }));

  return (
    <View
      className="relative flex-row rounded-xl bg-background-300 p-1"
      onLayout={e => setContainerWidth(e.nativeEvent.layout.width)}>
      {/* Sliding pill */}
      {containerWidth > 0 && (
        <Animated.View
          style={[
            sliderStyle,
            {
              top: 4,
              bottom: 4,
              left: 4,
              width: containerWidth / 2 - 8, // half width minus padding
            },
          ]}
          className="absolute rounded-lg bg-background-0"
        />
      )}

      <Pressable
        onPress={() => setMode('light')}
        className="flex-1 items-center justify-center p-3">
        <Text
          className={
            mode === 'light'
              ? 'font-bodyBold text-primary-500'
              : 'text-primary-300'
          }>
          Light
        </Text>
      </Pressable>

      <Pressable
        onPress={() => setMode('dark')}
        className="flex-1 items-center justify-center p-2">
        <Text
          className={
            mode === 'dark'
              ? 'font-bodyBold text-primary-500'
              : 'text-primary-300'
          }>
          Dark
        </Text>
      </Pressable>
    </View>
  );
}
