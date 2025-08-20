import React, { useState } from 'react';
import { View, Text } from '@/components/Themed';
import { Pressable } from 'react-native';

type DropdownItemProps = {
  label: string;
  onPress?: () => void;
};

export function DropdownItem({ label, onPress }: DropdownItemProps) {
  return (
    <Pressable onPress={onPress}>
      <Text className="px-2 font-bodySemiBold text-primary-500">{label}</Text>
    </Pressable>
  );
}

type CustomDropdownProps = {
  toggle: React.ReactNode;
  children: React.ReactElement<DropdownItemProps>[];
};

export default function CustomDropdown({
  toggle,
  children,
}: CustomDropdownProps) {
  const [open, setOpen] = useState(false);
  const toggleDropdown = () => {
    setOpen(prev => !prev);
    console.log('open');
  };
  return (
    <View className="relative">
      {React.isValidElement(toggle)
        ? React.cloneElement(toggle as React.ReactElement<any>, {
            onPress: toggleDropdown,
          })
        : toggle}
      {open && (
        <View className="absolute right-0 top-full min-w-40 rounded-2xl bg-background-0 p-3">
          {React.Children.map(children, (child, index) => (
            <View key={index}>
              {index > 0 && <View className="my-2 h-px bg-secondary-500/50" />}
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
