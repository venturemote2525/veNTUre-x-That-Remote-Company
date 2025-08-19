import { ThemedSafeAreaView, Text, TextInput, View } from '@/components/Themed';
import PasswordInput from '@/components/Auth/PasswordInput';
import Header from '@/components/Header';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Pressable } from 'react-native';
import { checkPasswordStrength, checkValidEmail } from '@/utils/auth/auth';

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
    login: '',
  });

  const handleSignup = async () => {
    setError({ email: '', password: '', reenter: '', login: '' });
    let hasError = false;
    if (!fields.email.trim()) {
      setError(prev => ({ ...prev, email: 'Please enter your email' }));
      hasError = true;
    } else if (!checkValidEmail(fields.email)) {
      setError(prev => ({ ...prev, email: 'Please enter valid email format' }));
      hasError = true;
    }
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
    if (!fields.reenter.trim()) {
      setError(prev => ({ ...prev, reenter: 'Please re-enter your password' }));
      hasError = true;
    } else if (fields.reenter !== fields.password) {
      setError(prev => ({
        ...prev,
        reenter: 'Please re-enter the same password',
      }));
      hasError = true;
    }

    if (hasError) return;

    try {
      setLoading(true);
      // TODO: Supabase sign up

      // Route to onboarding
      router.replace('/(auth)/onboarding');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemedSafeAreaView>
      <Header />
      <View className="flex-1 justify-center px-4">
        <View className="mb-8 items-center gap-1">
          <Text className="text-head1 font-heading">Sign Up</Text>
          <Text className="text-primary-200">Create a new account</Text>
        </View>
        {/* Email Login */}
        <View className="gap-4">
          <View className="gap-3">
            {/* Email Input */}
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
            {/* Password Input */}
            <View className="gap-1">
              <PasswordInput
                onChangeText={text => setFields({ ...fields, password: text })}
              />
              {error.password !== '' && (
                <Text className="text-error-500">{error.password}</Text>
              )}
            </View>
            {/* Re-Enter Password Input */}
            <View className="gap-1">
              <PasswordInput
                onChangeText={text => setFields({ ...fields, reenter: text })}
                placeholder="Re-enter your password"
              />
              {error.reenter !== '' && (
                <Text className="text-error-500">{error.reenter}</Text>
              )}
            </View>
          </View>
          <Pressable
            onPress={handleSignup}
            className="items-center rounded-2xl bg-secondary-500 py-3">
            <Text className="font-bodyBold text-xl text-background-500">
              {loading ? 'Creating new account...' : 'Sign Up'}
            </Text>
          </Pressable>
        </View>
        <View className="mt-4 flex-row justify-center">
          <Text>Have an account? </Text>
          <Pressable onPress={() => router.replace('/(auth)/login')}>
            <Text className="font-bodyBold text-secondary-500">Log In</Text>
          </Pressable>
        </View>
      </View>
    </ThemedSafeAreaView>
  );
}
