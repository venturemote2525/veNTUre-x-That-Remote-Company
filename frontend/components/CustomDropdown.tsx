// components/CustomDropdown.tsx
import React, { useState } from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import { BlurView } from 'expo-blur';
import { View, Text } from '@/components/Themed';
import { Colors, Shadows } from '@/constants/Colors';
import { AnimatedPressable, useFadeIn, useSlideIn } from '@/components/AnimatedComponents';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { IconDefinition } from '@fortawesome/fontawesome-svg-core';

type DropdownItemProps = {
  label: string;
  onPress?: () => void;
  icon?: IconDefinition;
  itemClassName?: any; // now accepts TextStyle
  itemTextClassName?: any; // now accepts TextStyle
};

export function DropdownItem({
  label,
  onPress,
  icon,
  itemClassName,
  itemTextClassName,
}: DropdownItemProps) {
  return (
    <AnimatedPressable
      onPress={onPress}
      scaleAmount={0.95}
      style={[styles.item, itemClassName]}
    >
      {icon && (
        <FontAwesomeIcon
          icon={icon}
          size={16}
          color={Colors.light.colors.primary[600]}
        />
      )}
      <Text style={[styles.itemText, itemTextClassName]}>{label}</Text>
    </AnimatedPressable>
  );
}

type CustomDropdownProps = {
  toggle: React.ReactNode;
  children: React.ReactElement<DropdownItemProps>[];
  menuClassName?: any;
  toggleClassName?: any;
  separator?: boolean;
  gap?: number;
  maxHeight?: number;
};

export default function CustomDropdown({
  toggle,
  children,
  separator,
  gap = 8,
  maxHeight = 200,
}: CustomDropdownProps) {
  const [open, setOpen] = useState(false);

  const toggleDropdown = () => setOpen(prev => !prev);

  // Animations
  const fadeStyle = useFadeIn(200);
  const slideStyle = useSlideIn(200, 'top');

  return (
    <View style={{ position: 'relative' }}>
      {React.isValidElement(toggle)
        ? React.cloneElement(toggle as React.ReactElement<any>, {
            onPress: toggleDropdown,
          })
        : toggle}

      {open && (
        <AnimatedPressable
          style={[styles.menuContainer, slideStyle, fadeStyle]}
        >
          <BlurView intensity={95} style={styles.blurContainer}>
            <ScrollView
              style={{ borderRadius: 12 }}
              contentContainerStyle={{ gap }}
              showsVerticalScrollIndicator={false}
            >
              {React.Children.map(children, (child, index) => (
                <View key={index}>
                  {separator && index > 0 && <View style={styles.separator} />}
                  {React.isValidElement(child)
                    ? React.cloneElement(child, {
                        onPress: () => {
                          child.props.onPress?.();
                          setOpen(false); // keep logic intact
                        },
                      })
                    : child}
                </View>
              ))}
            </ScrollView>
          </BlurView>
        </AnimatedPressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  item: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  itemText: {
    fontSize: 16,
    fontWeight: '500',
    color: Colors.light.colors.primary[600],
  },
  menuContainer: {
    position: 'absolute',
    right: 0,
    top: '100%',
    zIndex: 50,
    marginTop: 8,
    minWidth: 160,
    maxHeight: 200,
    borderRadius: 16,
    overflow: 'hidden',
    ...Shadows.large,
  },
  blurContainer: {
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: 8,
  },
  separator: {
    height: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.1)',
    marginVertical: 4,
  },
});
