import { Text, ThemedSafeAreaView, View } from '@/components/Themed';

export default function ProfileScreen() {
  return (
    <ThemedSafeAreaView>
      <View className="flex-1 items-center justify-center">
        <Text>Profile</Text>
      </View>
    </ThemedSafeAreaView>
  );
}
