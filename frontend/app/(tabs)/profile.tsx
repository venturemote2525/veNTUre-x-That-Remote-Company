import { CustomAlert } from '@/components/CustomAlert';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useAuth } from '@/context/AuthContext';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Pressable } from 'react-native';

export default function ProfileScreen() {
  const router = useRouter();
  const { profile } = useAuth();
  const [alertVisible, setAlertVisible] = useState(false);

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
      <View className="flex-1 justify-between px-4">
        <Text className="text-primary-500">{profile?.username}</Text>
        <Pressable onPress={handleLogout} className="button-rounded">
          <Text className="font-bodyBold text-background-0">Log out</Text>
        </Pressable>
      </View>
    </ThemedSafeAreaView>
  );
}
