import React, { useState } from 'react';
import { View, Text } from '@/components/Themed';
import { Pressable } from 'react-native';

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
  separator?: boolean;
};

export default function CustomDropdown({
  toggle,
  children,
  menuClassName,
  separator,
}: CustomDropdownProps) {
  const [open, setOpen] = useState(false);
  const toggleDropdown = () => {
    setOpen(prev => !prev);
  };
  return (
    <View className="relative">
      {React.isValidElement(toggle)
        ? React.cloneElement(toggle as React.ReactElement<any>, {
            onPress: toggleDropdown,
          })
        : toggle}
      {open && (
        <View
          className={`absolute right-0 top-full mt-2 ${menuClassName ?? ''}`}>
          {React.Children.map(children, (child, index) => (
            <View key={index}>
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
          ))}
        </View>
      )}
    </View>
  );
}
