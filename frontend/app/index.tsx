import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useRouter } from 'expo-router';
import { Pressable } from 'react-native';

export default function Index() {
  const router = useRouter();
  return (
    <ThemedSafeAreaView className="flex">
      <View className="flex-1 items-center justify-center">
        <Text className="font-heading text-title text-secondary-500">
          STRIDE
        </Text>
      </View>
      <View className="flex w-full gap-6 px-4 py-12">
        <Pressable
          onPress={() => router.push('/(auth)/signup')}
          className="button-white flex w-full">
          <Text className="text-lg font-bodyBold text-primary-500">
            Sign Up
          </Text>
        </Pressable>
        <Pressable
          onPress={() => router.push('/(auth)/login')}
          className="button flex w-full">
          <Text className="text-lg font-bodyBold text-background-500">
            Log In
          </Text>
        </Pressable>
      </View>
    </ThemedSafeAreaView>
  );
}
