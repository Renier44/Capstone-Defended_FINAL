import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Image,
    ActivityIndicator,
    Alert, 
    ScrollView,
    Platform,
    Modal // Import Modal component
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { MaterialIcons, Feather, Entypo, Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import * as ImagePicker from 'expo-image-picker';
import Constants from 'expo-constants';


// =======================================================
// 1. GLOBAL CONSTANTS
// =======================================================
const BRAND_BLUE = "#0057B7"; 
const ATTENTION_COLOR = "#FF6B6B"; 
const PRIMARY_ACTION_COLOR = "#FFD54F"; 
const BACKGROUND_COLOR = "#E8F7FF"; 
const NEUTRAL_TEXT = "#333333"; 
const INFO_BORDER_COLOR = "#77CDE0"; 
const CARD_BG = "#fff"; 
const LOGOUT_TEXT = '#FF5A5A'; 


// =======================================================
// 2. MODAL COMPONENT (Updated with unified button styles)
// =======================================================

const LogoutModal = ({ isVisible, onClose, onConfirm }) => {
    return (
        <Modal
            animationType="fade"
            transparent={true}
            visible={isVisible}
            onRequestClose={onClose}
        >
            <View style={modalStyles.centeredView}>
                <View style={modalStyles.modalView}>
                    <Text style={modalStyles.modalTitle}>Logout</Text>
                    <Text style={modalStyles.modalMessage}>are you sure you want to log out?</Text>
                    
                    <View style={modalStyles.buttonContainer}>
                        {/* Cancel Button - Unified look with press effect */}
                        <TouchableOpacity
                            style={[modalStyles.button, modalStyles.buttonCancel]}
                            onPress={onClose}
                            activeOpacity={0.7} // Press effect
                        >
                            <Text style={modalStyles.textCancel}>Cancel</Text>
                        </TouchableOpacity>

                        {/* Yes, Logout Button - Unified look with press effect */}
                        <TouchableOpacity
                            style={[modalStyles.button, modalStyles.buttonConfirm]}
                            onPress={onConfirm}
                            activeOpacity={0.7} // Press effect
                        >
                            <Text style={modalStyles.textConfirm}>Yes, Logout</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        </Modal>
    );
};

// =======================================================
// 3. MAIN PROFILE COMPONENT
// =======================================================

export default function Profile() {
    const router = useRouter();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [isLogoutModalVisible, setIsLogoutModalVisible] = useState(false);

    // Helper to normalize the user data structure
    const normalizeUser = (data) => ({
        ...data,
        name: data.name || data.first_name || '',
        lastName: data.lastName || data.last_name || '',
        email: data.email || data.username || '',
        profile_image: data.profile_image || null,
        first_name: data.first_name || data.name || '',
        last_name: data.last_name || data.lastName || '',
    });

    // Function to load profile from storage
    const loadUser = async () => {
        setLoading(true);
        try {
            const storedUserStr = await SecureStore.getItemAsync('userProfile');
            if (storedUserStr) {
                const storedUser = JSON.parse(storedUserStr);
                const normalizedUser = normalizeUser(storedUser);
                setUser(normalizedUser);
            } else {
                const token = await SecureStore.getItemAsync('userToken');
                if (!token) {
                    router.replace('/login');
                    return;
                }
                setUser(normalizeUser({ first_name: 'Guest', last_name: 'User', email: 'guest@example.com' }));
            }
        } catch (error) {
            console.error('Failed to load user profile:', error);
            setUser(normalizeUser({ first_name: 'Error', last_name: 'Loading', email: 'error@example.com' }));
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadUser();
    }, []);

    // Function to handle image picking and profile update (Unchanged)
    const updateProfileImage = async () => {
        if (!user) {
            Alert.alert('Error', 'User data not loaded yet.');
            return;
        }

        try {
            const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (permissionResult.status !== 'granted') {
                Alert.alert('Permission Denied', 'Permission to access photos is required!');
                return;
            }

            const pickerResult = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Images,
                allowsEditing: true,
                aspect: [1, 1],
                quality: 0.7,
            });

            if (!pickerResult.canceled) {
                const newImageUri = pickerResult.assets[0].uri;
                const updatedUser = { 
                    ...user,
                    profile_image: newImageUri 
                };
                setUser(updatedUser);
                await SecureStore.setItemAsync('userProfile', JSON.stringify(updatedUser));
                Alert.alert('Success', 'Profile image updated and saved locally.');
            }
        } catch (error) {
            console.error('Error picking or updating image:', error);
            Alert.alert('Error', 'Failed to update the profile image.');
        }
    };

    // FINAL LOGOUT LOGIC (Called by the modal)
    const confirmLogout = async () => {
        setIsLogoutModalVisible(false); // Close modal first
        await SecureStore.deleteItemAsync('userToken');
        router.replace('/login');
    };

    // OPEN MODAL HANDLER
    const handleLogoutPress = () => {
        setIsLogoutModalVisible(true);
    };


    if (loading) {
        return (
            <SafeAreaView style={[styles.container, styles.loadingContainer]}>
                <ActivityIndicator size="large" color={BRAND_BLUE} />
            </SafeAreaView>
        );
    }

    if (!user) return null;

    const displayName = `${user.first_name} ${user.last_name}`.trim() || 'User Profile';


    return (
        <SafeAreaView style={styles.container}>
            
            <View style={styles.fixedHeader}>
                <TouchableOpacity style={styles.backButton} onPress={() => {
                    if (router.canGoBack()) router.back();
                    else router.replace('/dashboard');
                }}>
                    <Ionicons name="arrow-back" size={24} color={BRAND_BLUE} />
                </TouchableOpacity>
                <Text style={styles.pageTitleFixed}>My Profile</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent}>
                
                <View style={[styles.card, styles.profileSection]}>
                    <TouchableOpacity 
                        style={styles.profilePicWrapper} 
                        onPress={updateProfileImage}
                    >
                        <Image
                            source={{
                                uri:
                                    user.profile_image
                                        ? user.profile_image
                                        : 'https://cdn-icons-png.flaticon.com/512/149/149071.png',
                            }}
                            style={styles.profilePic}
                        />
                        <TouchableOpacity 
                            style={styles.editIconWrapper} 
                            onPress={updateProfileImage}
                        >
                            <Feather name="edit-3" size={18} color="#fff" />
                        </TouchableOpacity>
                    </TouchableOpacity>
                    <Text style={styles.userName}>{displayName}</Text>
                    <Text style={styles.userEmail}>{user.email}</Text>
                </View>

                <View style={[styles.card, styles.menuContainer]}>
                    <TouchableOpacity
                        style={styles.menuItem}
                        onPress={() => router.push('/edit-profile')}
                    >
                        <MaterialIcons name="person-outline" size={24} color={BRAND_BLUE} />
                        <Text style={styles.menuText}>Edit Profile</Text>
                        <Entypo name="chevron-right" size={22} color={NEUTRAL_TEXT} />
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={styles.menuItem}
                        onPress={() => router.push('/privacy-policy')}
                    >
                        <MaterialIcons name="lock-outline" size={24} color={BRAND_BLUE} />
                        <Text style={styles.menuText}>Privacy Policy</Text>
                        <Entypo name="chevron-right" size={22} color={NEUTRAL_TEXT} />
                    </TouchableOpacity>

                    
                    {/* Calls the handler to open the modal */}
                    <TouchableOpacity
                        style={[styles.menuItem, styles.logoutItem]}
                        onPress={handleLogoutPress} 
                    >
                        <MaterialIcons name="logout" size={24} color={ATTENTION_COLOR} />
                        <Text style={[styles.menuText, { color: LOGOUT_TEXT }]}>Logout</Text>
                        <Entypo name="chevron-right" size={22} color={ATTENTION_COLOR} />
                    </TouchableOpacity>
                </View>
                
                <Text style={styles.versionText}>App Version 1.0.0</Text>

            </ScrollView>

            {/* MODAL RENDERED HERE */}
            <LogoutModal
                isVisible={isLogoutModalVisible}
                onClose={() => setIsLogoutModalVisible(false)}
                onConfirm={confirmLogout}
            />

        </SafeAreaView>
    );
}

// =======================================================
// 4. STYLES DEFINITIONS
// =======================================================

const CARD_ELEVATION = {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 3,
    elevation: 3,
};

const styles = StyleSheet.create({
    // --- Overall Layout ---
    container: {
        flex: 1,
        backgroundColor: BACKGROUND_COLOR, 
    },
    loadingContainer: {
        justifyContent: 'center',
        alignItems: 'center',
    },
    scrollContent: {
        padding: 20,
        paddingTop: 0, 
    },

    // --- Fixed Header ---
    fixedHeader: {
        height: Platform.OS === 'android' ? Constants.statusBarHeight + 50 : 60, 
        width: '100%',
        paddingHorizontal: 20,
        paddingBottom: 5, 
        backgroundColor: 'transparent', 
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center', 
    },
    pageTitleFixed: {
        fontSize: 25, 
        fontFamily: "Montserrat-VariableFont_wght",
        fontWeight: "800",
        color: BRAND_BLUE, 
        textAlign: "center",
        flex: 1, 
        marginTop:30,
    },
    backButton: {
        position: "absolute",
        // UPDATED: Decreased the value by 5 to move it up.
        top: Platform.OS === 'android' ? Constants.statusBarHeight + 5 : 5, 
        left: 20,
        zIndex: 10,
        padding: 10,
        backgroundColor: "#fff",
        borderRadius: 50, 
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 3.84,
        elevation: 5,
        
    },
    
    // --- Card Base Style ---
    card: {
        width: "100%", 
        backgroundColor: CARD_BG,
        borderRadius: 15,
        ...CARD_ELEVATION,
        marginBottom: 20,
        borderLeftWidth: 5,
        borderLeftColor: INFO_BORDER_COLOR, 
    },

    // --- Profile Section ---
    profileSection: {
        alignItems: 'center',
        paddingVertical: 30,
        paddingHorizontal: 20,
        backgroundColor: CARD_BG, 
        marginTop: 35,
        paddingLeft: 25, 
    },
    profilePicWrapper: {
        position: 'relative',
        marginBottom: 15,
    },
    profilePic: {
        width: 120, 
        height: 120,
        borderRadius: 60,
        backgroundColor: BACKGROUND_COLOR, 
        borderWidth: 4,
        borderColor: INFO_BORDER_COLOR, 
    },
    editIconWrapper: {
        position: 'absolute',
        bottom: 0,
        right: 0,
        backgroundColor: PRIMARY_ACTION_COLOR, 
        borderRadius: 20,
        padding: 8,
        borderWidth: 2,
        borderColor: CARD_BG, 
        ...CARD_ELEVATION,
    },
    userName: {
        fontSize: 26,
        fontWeight: '900',
        color: NEUTRAL_TEXT,
        fontFamily: "Montserrat-VariableFont_wght",
        marginBottom: 4,
    },
    userEmail: {
        fontSize: 16,
        color: '#666',
        fontFamily: "VarelaRound-Regular",
    },

    // --- Menu Options ---
    menuContainer: {
        paddingVertical: 10,
        paddingLeft: 20, 
    },
    menuItem: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 15,
        paddingHorizontal: 5, 
        borderBottomColor: BACKGROUND_COLOR, 
        borderBottomWidth: 1,
    },
    logoutItem: {
        borderBottomWidth: 0, 
    },
    menuText: {
        flex: 1,
        marginLeft: 15,
        fontSize: 16,
        color: NEUTRAL_TEXT,
        fontFamily: "VarelaRound-Regular",
    },
    
    // --- Footer ---
    versionText: {
        fontSize: 12,
        color: '#999',
        textAlign: 'center',
        marginTop: 30,
        fontFamily: "VarelaRound-Regular",
    }
});


// =======================================================
// 5. MODAL STYLES (Updated for unified button look)
// =======================================================

const modalStyles = StyleSheet.create({
    centeredView: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.4)',
    },
    modalView: {
        margin: 20,
        backgroundColor: CARD_BG,
        borderRadius: 20,
        padding: 35,
        alignItems: 'center',
        width: '80%',
        ...CARD_ELEVATION,
    },
    modalTitle: {
        fontSize: 22,
        fontWeight: '700',
        color: BRAND_BLUE,
        marginBottom: 10,
        fontFamily: "Montserrat-VariableFont_wght",
    },
    modalMessage: {
        marginBottom: 20,
        textAlign: 'center',
        fontSize: 15,
        color: NEUTRAL_TEXT,
        fontFamily: "VarelaRound-Regular",
    },
    buttonContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '100%',
    },
    button: {
        borderRadius: 12,
        padding: 12,
        width: '48%',
        alignItems: 'center',
        ...CARD_ELEVATION,
        elevation: 2,
        backgroundColor: PRIMARY_ACTION_COLOR, // Unified background color
    },
    // Left button: Cancel (Text color only)
    buttonCancel: {
        // Inherits background from `button`
    },
    textCancel: {
        color: BRAND_BLUE, 
        fontWeight: '600',
        fontSize: 15,
        fontFamily: "Montserrat-VariableFont_wght",
    },
    // Right button: Yes, Logout (Text color only)
    buttonConfirm: {
        // Inherits background from `button`
    },
    textConfirm: {
        color: BRAND_BLUE, // Unified text color for contrast
        fontWeight: '600',
        fontSize: 15,
        fontFamily: "Montserrat-VariableFont_wght",
    },
});