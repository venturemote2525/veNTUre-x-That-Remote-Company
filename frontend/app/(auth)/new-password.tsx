import { ThemedSafeAreaView, Text, View, TextInput } from '@/components/Themed';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { AnimatedPressable } from '@/components/AnimatedComponents';
import { supabase } from '@/lib/supabase';

export default function NewPassword() {
  const router = useRouter();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleResetPassword = async () => {
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    try {
      setLoading(true);
      const { error } = await supabase.auth.updateUser({
        password: password
      });

      if (error) throw error;

      // Password updated successfully
      router.replace('/(auth)/login');
    } catch (error) {
      console.error('Error resetting password:', error);
      setError('Failed to reset password. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemedSafeAreaView className="flex-1">
      <View className="flex-1 justify-center px-4">
        <View className="mb-8 items-center gap-1">
          <Text className="font-heading text-[50px] text-secondary-500">
            HealthSync
          </Text>
          <Text className="font-heading text-head2 text-primary-500">
            Set New Password
          </Text>
          <Text className="text-primary-200 text-center">
            Enter your new password below
          </Text>
        </View>

        <View className="gap-4">
          <View className="gap-3">
            <TextInput
              placeholder="New Password"
              secureTextEntry
              onChangeText={setPassword}
              className="h-14 items-center rounded-2xl bg-background-0 px-4"
            />
            <TextInput
              placeholder="Confirm New Password"
              secureTextEntry
              onChangeText={setConfirmPassword}
              className="h-14 items-center rounded-2xl bg-background-0 px-4"
            />
            {error !== '' && (
              <Text className="text-error-500 text-center">{error}</Text>
            )}
          </View>

          <AnimatedPressable onPress={handleResetPassword} className="button">
            <Text className="text-xl font-bodyBold text-background-500">
              {loading ? 'Resetting...' : 'Reset Password'}
            </Text>
          </AnimatedPressable>
        </View>
      </View>
    </ThemedSafeAreaView>
  );
}