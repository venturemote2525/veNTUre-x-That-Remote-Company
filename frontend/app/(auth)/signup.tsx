import { ThemedSafeAreaView, Text, TextInput, View } from '@/components/Themed';
import PasswordInput from '@/components/Auth/PasswordInput';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { checkPasswordStrength, checkValidEmail } from '@/utils/auth/auth';
import { checkUserExists, userSignup } from '@/utils/auth/api';
import { CustomAlert } from '@/components/CustomAlert';
import { AnimatedPressable } from '@/components/AnimatedComponents';

export default function SignUp() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [alert, setAlert] = useState({
    visible: false,
    title: '',
    message: '',
  });
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
      console.log('sign up');
      // Check if user exists
      const exists = await checkUserExists(fields.email.trim().toLowerCase());
      if (exists) {
        setAlert({
          visible: true,
          title: 'User exists',
          message: 'An account with this email already exists!',
        });
        return;
      }
      // Supabase sign up
      await userSignup(fields.email.trim().toLowerCase(), fields.password);
      setAlert({
        visible: true,
        title: 'Signup successful',
        message: 'Please check your email to verify your account!',
      });
    } catch (error: any) {
      const message =
        error?.message || error?.error_description || 'Unknown error occurred';

      console.log('signup error: ', error);
      setAlert({
        visible: true,
        title: 'Signup failed',
        message,
      });
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
          <Text className="text-primary-200">Create a new account</Text>
        </View>
        
        <View className="gap-4">
          <View className="gap-3">
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
            <View className="gap-1">
              <PasswordInput
                onChangeText={text => setFields({ ...fields, password: text })}
              />
              {error.password !== '' && (
                <Text className="text-error-500">{error.password}</Text>
              )}
            </View>
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
          <AnimatedPressable onPress={handleSignup} className="button">
            <Text className="text-xl font-bodyBold text-background-500">
              {loading ? 'Creating new account...' : 'Sign Up'}
            </Text>
          </AnimatedPressable>
        </View>

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
        
        <View className="mt-4 flex-row justify-center">
          <Text className="text-primary-500">Have an account? </Text>
          <AnimatedPressable onPress={() => router.replace('/(auth)/login')}>
            <Text className="font-bodyBold text-secondary-500">Log In</Text>
          </AnimatedPressable>
        </View>
      </View>

      <CustomAlert
        visible={alert.visible}
        title={alert.title}
        message={alert.message}
        onConfirm={() => {
          setAlert({ ...alert, visible: false });
        }}
      />
    </ThemedSafeAreaView>
  );
}