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
      <Text>{label}</Text>
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
  const toggleDropdown = () => setOpen(prev => !prev);
  return (
    <View>
      {React.isValidElement(toggle)
        ? React.cloneElement(toggle as React.ReactElement<any>, {
            onPress: toggleDropdown,
          })
        : toggle}
      {open && (
        <View>
          {React.Children.map(children, child =>
            React.isValidElement(child)
              ? React.cloneElement(child, {
                  onPress: () => {
                    child.props.onPress?.();
                    setOpen(false);
                  },
                })
              : child,
          )}
        </View>
      )}
    </View>
  );
}
