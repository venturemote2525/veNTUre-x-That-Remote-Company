import { useRef, useEffect, useState } from 'react';
import { View, Animated } from 'react-native';
import { Text } from '@/components/Themed';

type LoadingScreenProps = {
  text?: string;
};

export default function LoadingScreen({ text }: LoadingScreenProps) {
  const animations = [
    useRef(new Animated.Value(0)).current,
    useRef(new Animated.Value(0)).current,
    useRef(new Animated.Value(0)).current,
  ];
  const [dotCount, setDotCount] = useState(0);
  useEffect(() => {
    // cycle dotCount: 0 → 1 → 2 → 3 → 0 ...
    const interval = setInterval(() => {
      setDotCount(prev => (prev + 1) % 4);
    }, 500); // change every 500ms

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const createAnimation = (anim: Animated.Value, delay: number) => {
      return Animated.loop(
        Animated.sequence([
          Animated.timing(anim, {
            toValue: -15,
            duration: 300,
            delay,
            useNativeDriver: true,
          }),
          Animated.timing(anim, {
            toValue: 0,
            duration: 300,
            useNativeDriver: true,
          }),
        ]),
      );
    };

    const anims = animations.map((anim, i) => createAnimation(anim, i * 100));
    anims.forEach(a => a.start());
  }, []);

  return (
    <View className="flex-1 items-center justify-center gap-6">
      <View className="flex-row justify-between gap-2">
        {animations.map((anim, i) => (
          <Animated.View
            key={i}
            className="rounded-full bg-secondary-500"
            style={[
              { width: 12, height: 12 },
              { transform: [{ translateY: anim }] },
            ]}
          />
        ))}
      </View>
      {text && (
        <Text className="font-bodyBold text-secondary-500">
          {text + '.'.repeat(dotCount)}
        </Text>
      )}
    </View>
  );
}
