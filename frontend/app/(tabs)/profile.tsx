import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useAuth } from '@/context/AuthContext';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'expo-router';
import { Pressable, Image, ActivityIndicator } from 'react-native';
import { Link } from 'expo-router';
import { useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';

export default function ProfileScreen() {
  const router = useRouter();
  const { profile, user, profileLoading, loading, refreshProfile } = useAuth();

  const email = user?.email ?? undefined;
  const memberSinceIso =
    (profile as { created_at?: string } | null)?.created_at ?? user?.created_at;
  const memberSince =
    memberSinceIso ? new Date(memberSinceIso).toLocaleDateString() : undefined;

  const isBusy = loading || profileLoading;

  // === Avatar state ===
  const [photo, setPhoto] = useState<string | null>(
    (profile as any)?.avatar_url ?? null
  );
  const [uploading, setUploading] = useState(false);

  // === Logout ===
  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) console.log('Error signing out:', error);
  };

  // === Pick image from gallery ===
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
      base64: true,
    });
    if (!result.canceled && result.assets.length > 0) {
      const base64Data = result.assets[0].base64!;
      uploadPhoto(base64Data);
    }
  };

  // === Upload to Supabase Storage ===
  const uploadPhoto = async (base64Data: string) => {
    if (!user) return;
    try {
      setUploading(true);
      const fileName = `avatars/${user.id}.png`;

      const { error } = await supabase.storage
        .from('avatars')
        .upload(fileName, Buffer.from(base64Data, 'base64'), {
          contentType: 'image/png',
          upsert: true,
        });

      if (error) throw error;

      const { data: urlData } = supabase.storage.from('avatars').getPublicUrl(fileName);
      setPhoto(urlData.publicUrl);

      if (refreshProfile) await refreshProfile();
    } catch (err) {
      console.log('Error uploading photo:', err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <ThemedSafeAreaView>
      <View className="flex-1 px-6 py-10 bg-background-50">
        {/* Profile Header */}
        <View className="items-center mb-6">
        <Pressable
  onPress={pickImage}
  className="rounded-full overflow-hidden"
  style={{ width: 96, height: 96 }}
>
  <Image
    source={{
      uri:
        photo ??
        'https://ui-avatars.com/api/?name=' +
          encodeURIComponent(profile?.username || email || 'User'),
    }}
    style={{ width: 96, height: 96, borderRadius: 48 }}
  />
  {/* Overlay indicator in center */}
  <View
    style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: 96,
      height: 96,
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: 'rgba(0,0,0,0.2)', // subtle overlay
      borderRadius: 48,
    }}
  >
    {uploading ? (
      <ActivityIndicator color="#fff" size="small" />
    ) : (
      <Ionicons name="camera-outline" size={24} color="#fff" />
    )}
  </View>
</Pressable>


          <Text className="text-xl font-bodyBold text-primary-600 mt-2">
            {profile?.username || 'Guest User'}
          </Text>
          <Text className="text-sm text-gray-500">
            {isBusy ? 'Loading...' : email ?? 'No email'}
          </Text>
        </View>

        {/* Info Section */}
        <View className="space-y-4 bg-white rounded-2xl shadow-md p-6 mb-10">
          <View className="flex-row justify-between">
            <Text className="text-gray-500 font-body">Member since</Text>
            <Text className="font-bodyBold text-black">
              {isBusy ? 'Loading...' : memberSince ?? 'N/A'}
            </Text>
          </View>
          <View className="h-[1px] bg-gray-200" />
          <View className="flex-row justify-between">
            <Text className="text-gray-500 font-body">Status</Text>
            <Text className="font-bodyBold text-green-600">Active</Text>
          </View>
        </View>

        {/* Logout Button using Link */}
        <Link href="/" replace asChild>
          <Pressable
            onPress={handleLogout}
            className="bg-red-500 py-3 rounded-2xl shadow-md"
          >
            <Text className="font-bodyBold text-background-0 text-center">
              Log out
            </Text>
          </Pressable>
        </Link>
      </View>
    </ThemedSafeAreaView>
  );
}
