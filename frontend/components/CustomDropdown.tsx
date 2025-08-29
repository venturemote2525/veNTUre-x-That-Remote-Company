import React, { useState } from 'react';
import { View, Text } from '@/components/Themed';
import { Pressable, ScrollView } from 'react-native';

type DropdownItemProps = {
  label: string;
  onPress?: () => void;
  itemClassName?: string;
  itemTextClassName?: string;
};

export function DropdownItem({
  label,
  onPress,
  itemClassName,
  itemTextClassName,
}: DropdownItemProps) {
  return (
    <Pressable onPress={onPress} className={`${itemClassName ?? ''}`}>
      <Text className={`px-2 font-bodySemiBold ${itemTextClassName ?? ''}`}>
        {label}
      </Text>
    </Pressable>
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
  gap = 0,
  maxHeight = 200,
}: CustomDropdownProps) {
  const [open, setOpen] = useState(false);
  const toggleDropdown = () => {
    setOpen(prev => !prev);
  };
  return (
    <View className={`relative`}>
      {React.isValidElement(toggle)
        ? React.cloneElement(toggle as React.ReactElement<any>, {
            onPress: toggleDropdown,
          })
        : toggle}

      {open && (
        <View
          className={`absolute right-0 top-full z-50 mt-2 ${menuClassName ?? ''}`}
          style={{ elevation: 10, maxHeight: maxHeight }}>
          <ScrollView className="rounded-xl" contentContainerStyle={{ gap: gap }}>{React.Children.map(children, (child, index) => (
            <View key={index} className={``}>
              {separator && index > 0 && (
                <View className="my-2 h-px bg-secondary-500/50" />
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
          ))}</ScrollView>
        </View>
      )}
    </View>
  );
}
