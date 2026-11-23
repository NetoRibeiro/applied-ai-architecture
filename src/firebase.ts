import { initializeApp } from 'firebase/app';
import {
  GoogleAuthProvider,
  getAuth,
  signInWithPopup,
  signOut,
  type User,
} from 'firebase/auth';
import {
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocFromServer,
  getFirestore,
  serverTimestamp,
  setDoc,
  updateDoc,
  writeBatch,
} from 'firebase/firestore';
import firebaseConfig from '../firebase-applet-config.json';
import type {
  LogbookEntry,
  RegistrationMode,
  UserProfile,
  UserRole,
  UserStatus,
  WeekLogState,
} from './types';

export const ADMIN_BOOTSTRAP_EMAIL = 'dsjuvencio@gmail.com';

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app, firebaseConfig.firestoreDatabaseId);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

export enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

export interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId?: string | null;
    email?: string | null;
    emailVerified?: boolean | null;
    isAnonymous?: boolean | null;
    tenantId?: string | null;
    providerInfo?: {
      providerId?: string | null;
      email?: string | null;
    }[];
  };
}

export function handleFirestoreError(
  error: unknown,
  operationType: OperationType,
  path: string | null,
): never {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo:
        auth.currentUser?.providerData?.map((provider) => ({
          providerId: provider.providerId,
          email: provider.email,
        })) || [],
    },
    operationType,
    path,
  };
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

async function testConnection() {
  try {
    await getDocFromServer(doc(db, 'test', 'connection'));
  } catch (error) {
    if (error instanceof Error && error.message.includes('the client is offline')) {
      console.error('Please check your Firebase configuration.');
    }
  }
}

void testConnection();

export async function signInWithGoogle() {
  return signInWithPopup(auth, googleProvider);
}

export async function signOutUser() {
  return signOut(auth);
}

export function formatWeekDocId(week: number): string {
  return `week_${String(Math.trunc(week)).padStart(2, '0')}`;
}

export async function ensureUserProfile(user: User): Promise<void> {
  const userPath = `users/${user.uid}`;
  const userRef = doc(db, 'users', user.uid);
  const isOwnerAdmin = user.email === ADMIN_BOOTSTRAP_EMAIL && user.emailVerified;

  let existingSnap;
  try {
    existingSnap = await getDoc(userRef);
  } catch (error) {
    handleFirestoreError(error, OperationType.GET, userPath);
  }

  const cleanName = (user.displayName || user.email?.split('@')[0] || 'Estudante').slice(0, 120);
  const validPhoto =
    user.photoURL && user.photoURL.length > 0 && user.photoURL.length <= 500
      ? user.photoURL
      : undefined;

  if (!existingSnap.exists()) {
    let initialStatus: UserStatus = 'pending';
    let initialRole: UserRole = 'student';

    if (isOwnerAdmin) {
      initialStatus = 'approved';
      initialRole = 'admin';
    } else {
      try {
        const configSnap = await getDoc(doc(db, 'config', 'access'));
        if (configSnap.exists() && configSnap.data()?.registrationMode === 'open_authenticated') {
          initialStatus = 'approved';
        }
      } catch (error) {
        handleFirestoreError(error, OperationType.GET, 'config/access');
      }
    }

    const createPayload: Record<string, unknown> = {
      uid: user.uid,
      email: user.email || '',
      displayName: cleanName,
      role: initialRole,
      status: initialStatus,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
      lastLoginAt: serverTimestamp(),
    };
    if (validPhoto) {
      createPayload.photoURL = validPhoto;
    }

    try {
      await setDoc(userRef, createPayload);
    } catch (error) {
      handleFirestoreError(error, OperationType.CREATE, userPath);
    }

    if (isOwnerAdmin) {
      try {
        const configRef = doc(db, 'config', 'access');
        const configSnap = await getDoc(configRef);
        if (!configSnap.exists()) {
          await setDoc(configRef, {
            registrationMode: 'approval_required',
            updatedBy: user.uid,
            updatedAt: serverTimestamp(),
          });
        }
      } catch (error) {
        handleFirestoreError(error, OperationType.WRITE, 'config/access');
      }
    }
    return;
  }

  const updatePayload: Record<string, unknown> = {
    displayName: cleanName,
    lastLoginAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  };
  if (validPhoto) {
    updatePayload.photoURL = validPhoto;
  }

  try {
    await updateDoc(userRef, updatePayload);
  } catch (error) {
    handleFirestoreError(error, OperationType.UPDATE, userPath);
  }

  if (isOwnerAdmin) {
    try {
      const configRef = doc(db, 'config', 'access');
      const configSnap = await getDoc(configRef);
      if (!configSnap.exists()) {
        await setDoc(configRef, {
          registrationMode: 'approval_required',
          updatedBy: user.uid,
          updatedAt: serverTimestamp(),
        });
      }
    } catch (error) {
      handleFirestoreError(error, OperationType.WRITE, 'config/access');
    }
  }
}

export async function upsertWeekLog(
  uid: string,
  week: number,
  state: WeekLogState,
  knownExists: boolean,
): Promise<void> {
  const weekId = formatWeekDocId(week);
  const docPath = `users/${uid}/weekLogs/${weekId}`;
  const ref = doc(db, 'users', uid, 'weekLogs', weekId);

  const normalizedSteps = {
    reading: Boolean(state.steps.reading),
    self_check: Boolean(state.steps.self_check),
    exercise: Boolean(state.steps.exercise),
    gabarito: Boolean(state.steps.gabarito),
    deliverable: Boolean(state.steps.deliverable),
    commit: Boolean(state.steps.commit),
  };

  let existsInDb = knownExists;
  if (!existsInDb) {
    try {
      const snap = await getDoc(ref);
      existsInDb = snap.exists();
    } catch (error) {
      handleFirestoreError(error, OperationType.GET, docPath);
    }
  }

  if (!existsInDb) {
    try {
      await setDoc(ref, {
        ownerUid: uid,
        week: Math.trunc(week),
        steps: normalizedSteps,
        hoursSpent: Math.max(0, Math.min(500, Number(state.hoursSpent) || 0)),
        lastUpdated: (state.lastUpdated || new Date().toISOString().slice(0, 10)).slice(0, 32),
        deliverableSummary: (state.deliverableSummary || '').slice(0, 1000),
        notes: (state.notes || '').slice(0, 10000),
        blockers: (state.blockers || '').slice(0, 5000),
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });
    } catch (error) {
      handleFirestoreError(error, OperationType.CREATE, docPath);
    }
    return;
  }

  try {
    await updateDoc(ref, {
      steps: normalizedSteps,
      hoursSpent: Math.max(0, Math.min(500, Number(state.hoursSpent) || 0)),
      lastUpdated: (state.lastUpdated || new Date().toISOString().slice(0, 10)).slice(0, 32),
      deliverableSummary: (state.deliverableSummary || '').slice(0, 1000),
      notes: (state.notes || '').slice(0, 10000),
      blockers: (state.blockers || '').slice(0, 5000),
      updatedAt: serverTimestamp(),
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.UPDATE, docPath);
  }
}

export async function createLogEntry(
  uid: string,
  entry: Omit<LogbookEntry, 'id'>,
): Promise<string> {
  const colRef = collection(db, 'users', uid, 'logEntries');
  const newRef = doc(colRef);
  const docPath = `users/${uid}/logEntries/${newRef.id}`;

  const payload: Record<string, unknown> = {
    ownerUid: uid,
    date: entry.date.slice(0, 32),
    week: Math.trunc(entry.week),
    title: entry.title.trim().slice(0, 200),
    hours: Math.max(0, Math.min(100, Number(entry.hours) || 0)),
    notes: entry.notes.trim().slice(0, 10000),
    blockers: (entry.blockers || '').trim().slice(0, 3000),
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  };

  if (entry.commandSnippet && entry.commandSnippet.trim().length > 0) {
    payload.commandSnippet = entry.commandSnippet.trim().slice(0, 4000);
  }

  try {
    await setDoc(newRef, payload);
    return newRef.id;
  } catch (error) {
    handleFirestoreError(error, OperationType.CREATE, docPath);
  }
}

export async function removeLogEntry(uid: string, entryId: string): Promise<void> {
  const docPath = `users/${uid}/logEntries/${entryId}`;
  try {
    await deleteDoc(doc(db, 'users', uid, 'logEntries', entryId));
  } catch (error) {
    handleFirestoreError(error, OperationType.DELETE, docPath);
  }
}

export async function saveGovernanceDraft(
  uid: string,
  kind: 'adr' | 'rfc',
  content: string,
): Promise<void> {
  const docPath = `users/${uid}/drafts/${kind}`;
  const ref = doc(db, 'users', uid, 'drafts', kind);
  const safeContent = content.slice(0, 50000);

  let snap;
  try {
    snap = await getDoc(ref);
  } catch (error) {
    handleFirestoreError(error, OperationType.GET, docPath);
  }

  if (!snap.exists()) {
    try {
      await setDoc(ref, {
        ownerUid: uid,
        kind,
        content: safeContent,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });
    } catch (error) {
      handleFirestoreError(error, OperationType.CREATE, docPath);
    }
    return;
  }

  try {
    await updateDoc(ref, {
      content: safeContent,
      updatedAt: serverTimestamp(),
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.UPDATE, docPath);
  }
}

export async function updateAccessPolicy(
  adminUid: string,
  registrationMode: RegistrationMode,
): Promise<void> {
  const configPath = 'config/access';
  const ref = doc(db, 'config', 'access');

  let snap;
  try {
    snap = await getDoc(ref);
  } catch (error) {
    handleFirestoreError(error, OperationType.GET, configPath);
  }

  if (!snap.exists()) {
    try {
      await setDoc(ref, {
        registrationMode,
        updatedBy: adminUid,
        updatedAt: serverTimestamp(),
      });
    } catch (error) {
      handleFirestoreError(error, OperationType.CREATE, configPath);
    }
    return;
  }

  try {
    await updateDoc(ref, {
      registrationMode,
      updatedBy: adminUid,
      updatedAt: serverTimestamp(),
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.UPDATE, configPath);
  }
}

export async function adminUpdateUser(
  targetUser: UserProfile,
  patch: { status?: UserStatus; role?: UserRole },
  adminUid: string,
  canManageAdminMarkers: boolean,
): Promise<void> {
  const userPath = `users/${targetUser.uid}`;
  const nextStatus = patch.status ?? targetUser.status;
  const nextRole = patch.role ?? targetUser.role;

  const batch = writeBatch(db);
  const userUpdate: Record<string, unknown> = {
    updatedAt: serverTimestamp(),
  };
  if (patch.status !== undefined) {
    userUpdate.status = nextStatus;
  }
  if (patch.role !== undefined) {
    userUpdate.role = nextRole;
  }
  batch.update(doc(db, 'users', targetUser.uid), userUpdate);

  if (canManageAdminMarkers && patch.role !== undefined && patch.role !== targetUser.role) {
    const adminRef = doc(db, 'admins', targetUser.uid);
    if (nextRole === 'admin') {
      batch.set(adminRef, {
        uid: targetUser.uid,
        email: targetUser.email,
        grantedBy: adminUid,
        createdAt: serverTimestamp(),
      });
    } else {
      batch.delete(adminRef);
    }
  }

  try {
    await batch.commit();
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, userPath);
  }
}

export async function adminDeleteUser(targetUid: string): Promise<void> {
  const userPath = `users/${targetUid}`;
  try {
    await deleteDoc(doc(db, 'users', targetUid));
  } catch (error) {
    handleFirestoreError(error, OperationType.DELETE, userPath);
  }
}

export async function migrateLocalProgressToCloud(
  uid: string,
  localWeekLogs: Record<number, WeekLogState>,
  localEntries: LogbookEntry[],
): Promise<number> {
  let count = 0;

  for (const [weekKey, state] of Object.entries(localWeekLogs)) {
    const weekNum = Number(weekKey);
    if (!Number.isInteger(weekNum) || weekNum < 1 || weekNum > 32) continue;
    await upsertWeekLog(uid, weekNum, state, false);
    count += 1;
  }

  for (const entry of localEntries) {
    if (!entry.title?.trim() || !entry.notes?.trim()) continue;
    await createLogEntry(uid, {
      date: entry.date || new Date().toISOString().slice(0, 10),
      week: Math.min(32, Math.max(1, Math.trunc(entry.week || 1))),
      title: entry.title,
      hours: Number(entry.hours) || 0,
      notes: entry.notes,
      blockers: entry.blockers || '',
      commandSnippet: entry.commandSnippet,
    });
    count += 1;
  }

  return count;
}
