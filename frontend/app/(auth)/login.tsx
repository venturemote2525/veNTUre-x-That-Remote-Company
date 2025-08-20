import PasswordInput from '@/components/Auth/PasswordInput';
import Header from '@/components/Header';
import { ThemedSafeAreaView, Text, View, TextInput } from '@/components/Themed';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Pressable } from 'react-native';

export default function LogIn() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [fields, setFields] = useState({
    email: '',
    password: '',
  });
  const [error, setError] = useState({
    email: '',
    password: '',
    login: '',
  });

  const handleLogin = async () => {
    setError({ email: '', password: '', login: '' });
    let hasError = false;
    if (!fields.email.trim()) {
      setError(prev => ({ ...prev, email: 'Please enter your email' }));
      hasError = true;
    }
    if (!fields.password.trim()) {
      setError(prev => ({ ...prev, password: 'Please enter your password' }));
      hasError = true;
    }
    router.push('/(tabs)');
    if (hasError) return;

    try {
      setLoading(true);
      // TODO: Supabase login
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemedSafeAreaView>
      <Header />
      <View className="flex-1 justify-center px-4">
        <View className="mb-8 items-center gap-1">
          <Text className="font-heading text-head1 text-primary-500">
            Log In
          </Text>
          <Text className="text-primary-200">
            Log in to your account via email
          </Text>
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
          </View>
          <Pressable
            onPress={handleLogin}
            className="items-center rounded-2xl bg-secondary-500 py-3">
            <Text className="text-xl font-bodyBold text-background-500">
              {loading ? 'Logging in...' : 'Log In'}
            </Text>
          </Pressable>
        </View>

        <View className="my-6 flex-row items-center">
          <View className="h-[1px] flex-1 bg-primary-300" />
          <Text className="mx-3 text-primary-300">
            Sign in with social media
          </Text>
          <View className="h-[1px] flex-1 bg-primary-300" />
        </View>

        {/* Social Login */}
        <View>
          <Pressable
            onPress={() => console.log('Google login')}
            className="items-center rounded-2xl bg-background-0 py-3">
            <Text className="text-xl font-bodyBold text-primary-500">
              Log in with Google
            </Text>
          </Pressable>
        </View>

        <View className="mt-4 flex-row justify-center">
          <Text className="text-primary-500">Not a member? </Text>
          <Pressable onPress={() => router.replace('/(auth)/signup')}>
            <Text className="font-bodyBold text-secondary-500">
              Create a new account
            </Text>
          </Pressable>
        </View>
      </View>
    </ThemedSafeAreaView>
  );
}
