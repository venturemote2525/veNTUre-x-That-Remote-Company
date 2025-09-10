// Replace your components/Auth/PasswordInput.tsx with this enhanced version
import React, { useState } from 'react';
import { TextInputProps } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { View, TextInput } from '@/components/Themed';
import { EyeIcon, EyeOffIcon, Icon } from '@/components/ui/icon';
import { Colors, Shadows } from '@/constants/Colors';
import { AnimatedPressable } from '@/components/AnimatedComponents';

type PasswordInputProps = TextInputProps & { 
  placeholder?: string;
  style?: any;
};

export default function PasswordInput({
  placeholder = 'Enter your password',
  value,
  onChangeText,
  style,
  ...props
}: PasswordInputProps) {
  const [visible, setVisible] = useState(false);

  return (
    <View style={[
      {
        height: 56,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderRadius: 16,
        backgroundColor: 'white',
        paddingHorizontal: 16,
        ...Shadows.small,
      },
      style
    ]}>
      <TextInput
        style={{
          flex: 1,
          backgroundColor: 'transparent',
          fontSize: 16,
        }}
        secureTextEntry={!visible}
        placeholder={placeholder}
        value={value}
        onChangeText={onChangeText}
        {...props}
      />
      
      <AnimatedPressable 
        onPress={() => setVisible(!visible)}
        scaleAmount={0.9}
      >
        <LinearGradient
          colors={visible ? Colors.light.gradients.primary : Colors.light.gradients.secondary}
          style={{
            width: 32,
            height: 32,
            borderRadius: 16,
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <Icon 
            as={visible ? EyeIcon : EyeOffIcon} 
            size="sm"
            style={{ color: 'white' }}
          />
        </LinearGradient>
      </AnimatedPressable>
    </View>
  );
}