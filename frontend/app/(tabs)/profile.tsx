import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useAuth } from '@/context/AuthContext';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'expo-router';
import { Pressable } from 'react-native';

export default function ProfileScreen() {
  const router = useRouter();
  const { profile } = useAuth();
  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      console.log(error);
      router.dismissAll();
    } catch (error) {
      console.log('Error signing out: ', error);
    }
  };
  return (
    <ThemedSafeAreaView>
      <View className="flex-1 items-center justify-center">
        <Text>Profile</Text>
        <Pressable onPress={handleLogout} className="button-rounded">
          <Text>Log out</Text>
        </Pressable>
      </View>
    </ThemedSafeAreaView>
  );
}
