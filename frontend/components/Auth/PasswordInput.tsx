import { View, TextInput } from '@/components/Themed';
import { EyeIcon, EyeOffIcon, Icon } from '@/components/ui/icon';
import { useState } from 'react';
import { Pressable, TextInputProps } from 'react-native';

type PasswordInputProps = TextInputProps & { placeholder?: string };

export default function PasswordInput({
  placeholder = 'Enter your password',
  value,
  onChangeText,
  ...props
}: PasswordInputProps) {
  const [visible, setVisible] = useState(false);
  return (
    <View className="h-14 flex-row items-center justify-between rounded-2xl bg-background-0 px-4">
      <TextInput
        className="flex-1"
        secureTextEntry={!visible}
        placeholder={placeholder}
        value={value}
        onChangeText={onChangeText}
        {...props}
      />
      <Pressable onPress={() => setVisible(!visible)}>
        <Icon as={visible ? EyeIcon : EyeOffIcon} />
      </Pressable>
    </View>
  );
}
