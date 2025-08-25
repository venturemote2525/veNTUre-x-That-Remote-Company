import { View } from '@/components/Themed';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors } from '@/constants/Colors';
import { BottomTabBarProps } from '@react-navigation/bottom-tabs';
import { Pressable, useColorScheme, Animated } from 'react-native';
import { useEffect, useRef } from 'react';

export default function TabBar({
  state,
  descriptors,
  navigation,
}: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  const rawScheme = useColorScheme();
  const scheme: 'light' | 'dark' = rawScheme === 'dark' ? 'dark' : 'light';

  // One Animated.Value per tab
  const scaleValues = useRef(
    state.routes.map(
      (_, index: number) => new Animated.Value(state.index === index ? 1.2 : 1),
    ),
  ).current;

  useEffect(() => {
    state.routes.forEach((_, index: number) => {
      const isFocused = state.index === index;

      Animated.spring(scaleValues[index], {
        toValue: isFocused ? 1.2 : 1,
        speed: 20,
        bounciness: 12,
        useNativeDriver: true,
      }).start();
    });
  }, [state.index, scaleValues, state.routes]);

  return (
    <View
      style={{ paddingBottom: insets.bottom }}
      className="flex-row bg-background-0">
      {state.routes.map((route, index) => {
        const descriptor = descriptors[route.key];
        if (!descriptor) return null;
        const { options } = descriptor;
        const isFocused = state.index === index;
        const scale = scaleValues[index];

        const onPress = () => {
          const event = navigation.emit({
            type: 'tabPress',
            target: route.key,
            canPreventDefault: true,
          });
          if (!isFocused && !event.defaultPrevented) {
            navigation.navigate(route.name, route.params);
          }
        };

        const onLongPress = () => {
          navigation.emit({ type: 'tabLongPress', target: route.key });
        };

        return (
          <Pressable
            key={index}
            accessibilityState={isFocused ? { selected: true } : {}}
            onPress={onPress}
            onLongPress={onLongPress}
            style={{ flex: 1, alignItems: 'center', padding: 16 }}>
            <Animated.View style={{ transform: [{ scale }] }}>
              {options.tabBarIcon?.({
                focused: isFocused,
                color: isFocused
                  ? Colors[scheme].secondary
                  : Colors[scheme].primary,
                size: 24,
              })}
            </Animated.View>
          </Pressable>
        );
      })}
    </View>
  );
}
