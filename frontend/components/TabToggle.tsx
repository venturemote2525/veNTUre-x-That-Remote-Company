import { Pressable, Animated, Dimensions } from 'react-native';
import { View, Text } from '@/components/Themed';
import { useEffect, useRef } from 'react';
import {toUpperCase} from "@/utils/formatString";

interface TabToggleProps {
  tabs: string[];
  selectedTab: string;
  onTabChange: (tab: string) => void;
}

export default function TabToggle({
  tabs,
  selectedTab,
  onTabChange,
}: TabToggleProps) {
  const underlineAnim = useRef(new Animated.Value(0)).current;
  const screenWidth = Dimensions.get('window').width;
  const tabWidth = screenWidth / tabs.length;

  const moveUnderline = (index: number) => {
    Animated.spring(underlineAnim, {
      toValue: index * tabWidth,
      useNativeDriver: true,
      bounciness: 10,
    }).start();
  };

  useEffect(() => {
    const index = tabs.indexOf(selectedTab);
    moveUnderline(index);
  }, [selectedTab]);

  return (
    <View className="relative flex-row">
      {tabs.map(tab => (
        <Pressable
          key={tab}
          className="flex-1 py-3"
          onPress={() => onTabChange(tab)}>
          <Text
            className={`text-center font-bodyBold text-body2 ${
              selectedTab === tab ? 'text-secondary-500' : 'text-primary-300'
            }`}>
            {toUpperCase(tab)}
          </Text>
        </Pressable>
      ))}

      <Animated.View
        className="absolute bottom-0 left-0 h-1.5 rounded-full bg-secondary-500"
        style={{
          width: tabWidth,
          transform: [{ translateX: underlineAnim }],
        }}
      />
    </View>
  );
}
