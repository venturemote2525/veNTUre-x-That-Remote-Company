// components/CustomDropdown.tsx
import React, { useState } from 'react';
import { ScrollView } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { View, Text } from '@/components/Themed';
import { Colors, Shadows, Animations } from '@/constants/Colors';
import { AnimatedPressable } from '@/components/AnimatedComponents';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { IconDefinition } from '@fortawesome/fontawesome-svg-core';

type DropdownItemProps = {
  label: string;
  onPress?: () => void;
  icon?: IconDefinition; // Add this line
  itemClassName?: string;
  itemTextClassName?: string;
};

export function DropdownItem({
  label,
  onPress,
  icon, // Add this
  itemClassName,
  itemTextClassName,
}: DropdownItemProps) {
  return (
    <AnimatedPressable onPress={onPress} scaleAmount={0.95}>
      <View style={{
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 12,
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        flexDirection: 'row', // Add this
        alignItems: 'center', // Add this
        gap: 12, // Add this for spacing between icon and text
      }}>
        {icon && ( // Add this to render the icon if provided
          <FontAwesomeIcon
            icon={icon}
            size={16}
            color={Colors.light.colors.primary[600]}
          />
        )}
        <Text style={{
          fontSize: 16,
          fontWeight: '500',
          color: Colors.light.colors.primary[600],
        }}>
          {label}
        </Text>
      </View>
    </AnimatedPressable>
  );
}

type CustomDropdownProps = {
  toggle: React.ReactNode;
  children: React.ReactElement<DropdownItemProps>[];
  menuClassName?: string;
  toggleClassName?: string;
  separator?: boolean;
  gap?: number;
  maxHeight?: number;
};

export default function CustomDropdown({
  toggle,
  children,
  menuClassName,
  separator,
  gap = 8,
  maxHeight = 200,
}: CustomDropdownProps) {
  const [open, setOpen] = useState(false);
  const scale = useSharedValue(0.8);
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(-10);

  const toggleDropdown = () => {
    setOpen(prev => {
      const newOpen = !prev;
      
      if (newOpen) {
        // Opening animation
        scale.value = withSpring(1, Animations.bounce);
        opacity.value = withTiming(1, { duration: 200 });
        translateY.value = withSpring(0, Animations.spring);
      } else {
        // Closing animation
        scale.value = withTiming(0.8, { duration: 150 });
        opacity.value = withTiming(0, { duration: 150 });
        translateY.value = withTiming(-10, { duration: 150 });
      }
      
      return newOpen;
    });
  };

  const menuAnimatedStyle = useAnimatedStyle(() => ({
    transform: [
      { scale: scale.value },
      { translateY: translateY.value },
    ],
    opacity: opacity.value,
  }));

  return (
    <View style={{ position: 'relative' }}>
      {React.isValidElement(toggle)
        ? React.cloneElement(toggle as React.ReactElement<any>, {
            onPress: toggleDropdown,
          })
        : toggle}

      {open && (
        <Animated.View
          style={[
            menuAnimatedStyle,
            {
              position: 'absolute',
              right: 0,
              top: '100%',
              zIndex: 50,
              marginTop: 8,
              minWidth: 160,
              maxHeight: maxHeight,
              borderRadius: 16,
              overflow: 'hidden',
              ...Shadows.large,
            },
          ]}
        >
          <BlurView
            intensity={95}
            style={{
              borderRadius: 16,
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              padding: 8,
            }}
          >
            <ScrollView 
              style={{ borderRadius: 12 }} 
              contentContainerStyle={{ gap }}
              showsVerticalScrollIndicator={false}
            >
              {React.Children.map(children, (child, index) => (
                <View key={index}>
                  {separator && index > 0 && (
                    <View style={{
                      height: 1,
                      backgroundColor: 'rgba(0, 0, 0, 0.1)',
                      marginVertical: 4,
                    }} />
                  )}
                  {React.isValidElement(child)
                    ? React.cloneElement(child, {
                        onPress: () => {
                          child.props.onPress?.();
                          setOpen(false);
                        },
                      })
                    : child}
                </View>
              ))}
            </ScrollView>
          </BlurView>
        </Animated.View>
      )}
    </View>
  );
}