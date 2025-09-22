import PasswordInput from '@/components/Auth/PasswordInput';
import { ThemedSafeAreaView, Text, View, TextInput } from '@/components/Themed';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { AnimatedPressable } from '@/components/AnimatedComponents';
import { checkPasswordStrength, checkValidEmail } from '@/utils/auth/auth';
import { checkUserExists, userSignup } from '@/utils/auth/api';

export default function SignUp() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [fields, setFields] = useState({
    email: '',
    password: '',
    reenter: '',
  });
  const [error, setError] = useState({
    email: '',
    password: '',
    reenter: '',
    signup: '',
  });

  const handleSignup = async () => {
    setError({ email: '', password: '', reenter: '', signup: '' });
    let hasError = false;

    // Email check
    if (!fields.email.trim()) {
      setError(prev => ({ ...prev, email: 'Please enter your email' }));
      hasError = true;
    } else if (!checkValidEmail(fields.email)) {
      setError(prev => ({ ...prev, email: 'Please enter valid email format' }));
      hasError = true;
    }

    // Password check
    if (!fields.password.trim()) {
      setError(prev => ({ ...prev, password: 'Please enter your password' }));
      hasError = true;
    } else {
      const strength = checkPasswordStrength(fields.password);
      if (strength !== 'strong') {
        setError(prev => ({ ...prev, password: strength }));
        hasError = true;
      }
    }

    // Re-enter password check
    if (!fields.reenter.trim()) {
      setError(prev => ({ ...prev, reenter: 'Please re-enter your password' }));
      hasError = true;
    } else if (fields.reenter !== fields.password) {
      setError(prev => ({
        ...prev,
        reenter: 'Passwords do not match',
      }));
      hasError = true;
    }

    if (hasError) return;

    try {
      setLoading(true);
      // Check if user already exists
      const exists = await checkUserExists(fields.email.trim().toLowerCase());
      if (exists) {
        setError(prev => ({
          ...prev,
          signup: 'An account with this email already exists',
        }));
        return;
      }
      // Sign up
      await userSignup(fields.email.trim().toLowerCase(), fields.password);
      router.replace('/(auth)/login');
    } catch (error: any) {
      console.log('Signup failed: ', error);
      setError(prev => ({
        ...prev,
        signup: error?.message || 'Unknown error occurred',
      }));
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
            Sign Up
          </Text>
          <Text className="text-primary-200">
            Create a new account via email
          </Text>
        </View>

        <View className="gap-4">
          <View className="gap-3">
            {/* Email */}
            <View className="gap-1">
              <TextInput
                placeholder="Enter your email"
                onChangeText={text => setFields({ ...fields, email: text })}
                className="h-14 items-center rounded-2xl bg-background-0 px-4"
              />
              {error.email !== '' && (
                <Text className="text-error-500">{error.email}</Text>
              )}
            </View>

            {/* Password */}
            <View className="gap-1">
              <PasswordInput
                onChangeText={text => setFields({ ...fields, password: text })}
              />
              {error.password !== '' && (
                <Text className="text-error-500">{error.password}</Text>
              )}
            </View>

            {/* Re-enter Password */}
            <View className="gap-1">
              <PasswordInput
                placeholder="Re-enter your password"
                onChangeText={text => setFields({ ...fields, reenter: text })}
              />
              {error.reenter !== '' && (
                <Text className="text-error-500">{error.reenter}</Text>
              )}
            </View>
          </View>

          {/* Sign Up Button */}
          <AnimatedPressable onPress={handleSignup} className="button">
            <Text className="text-xl font-bodyBold text-background-500">
              {loading ? 'Creating account...' : 'Sign Up'}
            </Text>
          </AnimatedPressable>
        </View>

        {/* Social Signup */}
        <View className="my-6 flex-row items-center">
          <View className="h-[1px] flex-1 bg-primary-300" />
          <Text className="mx-3 text-primary-300">
            Sign up with social media
          </Text>
          <View className="h-[1px] flex-1 bg-primary-300" />
        </View>

        <View>
          <AnimatedPressable
            onPress={() => console.log('Google signup')}
            className="button-white">
            <Text className="text-xl font-bodyBold text-primary-500">
              Sign up with Google
            </Text>
          </AnimatedPressable>
        </View>

        {/* Already have account */}
        <View className="mt-4 flex-row justify-center">
          <Text className="text-primary-500">Already have an account? </Text>
          <AnimatedPressable onPress={() => router.replace('/(auth)/login')}>
            <Text className="font-bodyBold text-secondary-500">Log In</Text>
          </AnimatedPressable>
        </View>

        {/* Signup Error */}
        {error.signup !== '' && (
          <View className="mt-4 items-center">
            <Text className="text-error-500">{error.signup}</Text>
          </View>
        )}
      </View>
    </ThemedSafeAreaView>
  );
}
