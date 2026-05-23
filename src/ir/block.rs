use crate::ir::armlet::LINK_NONE;
use crate::ir::{Armlet, Op, ValueRef};

#[derive(Clone, Copy, Debug)]
pub enum Terminal {
    Invalid,
    LinkBlock { next_pc: u64 },
    DirectBranch { target_pc: u64, link: bool },
    ConditionalBranch { cond_nzcv: ValueRef, cond_code: u8, taken_pc: u64, not_taken_pc: u64 },
    CompareBranchZero { value: ValueRef, inverse: bool, taken_pc: u64, not_taken_pc: u64 },
    TestBranchBit { value: ValueRef, bit: u8, inverse: bool, taken_pc: u64, not_taken_pc: u64 },
    IndirectBranch { target: ValueRef, link: bool, is_ret: bool },
    Exception { kind: ExceptionKind, imm: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExceptionKind {
    Svc, Brk, Hvc, UnknownInst,
}

pub struct Block {
    pub terminal: Terminal,
    pub code: Vec<Armlet>,
    pub start_pc: u64,
    pub end_pc: u64,
    pub head: u32,
    pub tail: u32,
    pub cycles: u32,
}

impl Block {
    pub const INITIAL_CAPACITY: usize = 128;
    pub const MAX_NODES: usize = 65_536;

    pub fn new(start_pc: u64) -> Self {
        Self {
            terminal: Terminal::Invalid,
            code: Vec::with_capacity(Self::INITIAL_CAPACITY),
            start_pc,
            end_pc: start_pc,
            head: LINK_NONE,
            tail: LINK_NONE,
            cycles: 0,
        }
    }

    pub fn reset(&mut self, start_pc: u64) {
        self.code.clear();
        self.terminal = Terminal::Invalid;
        self.start_pc = start_pc;
        self.end_pc = start_pc;
        self.head = LINK_NONE;
        self.tail = LINK_NONE;
        self.cycles = 0;
    }

    #[inline]
    pub fn push(&mut self, mut armlet: Armlet) -> ValueRef {
        let idx = self.code.len() as u32;
        debug_assert!((idx as usize) < Self::MAX_NODES, "block exceeded MAX_NODES");
        armlet.prev = self.tail;
        armlet.next = LINK_NONE;
        self.code.push(armlet);
        if self.tail != LINK_NONE {
            self.code[self.tail as usize].next = idx;
        } else {
            self.head = idx;
        }
        self.tail = idx;
        ValueRef::new(idx)
    }

    #[inline]
    pub fn unlink(&mut self, v: ValueRef) {
        let idx = v.idx();
        let (prev, next) = {
            let n = &self.code[idx as usize];
            (n.prev, n.next)
        };
        if prev != LINK_NONE { self.code[prev as usize].next = next; }
        else                 { self.head = next; }
        if next != LINK_NONE { self.code[next as usize].prev = prev; }
        else                 { self.tail = prev; }
        let n = &mut self.code[idx as usize];
        n.prev = LINK_NONE;
        n.next = LINK_NONE;
        n.op = Op::Void;
        n.args = [ValueRef::NONE; 4];
    }

    #[inline] pub fn len(&self)      -> usize { self.code.len() }
    #[inline] pub fn is_empty(&self) -> bool  { self.head == LINK_NONE }

    #[inline]
    pub fn head_vr(&self) -> Option<ValueRef> {
        (self.head != LINK_NONE).then(|| ValueRef::new(self.head))
    }

    #[inline]
    pub fn tail_vr(&self) -> Option<ValueRef> {
        (self.tail != LINK_NONE).then(|| ValueRef::new(self.tail))
    }

    #[inline]
    pub fn next_of(&self, v: ValueRef) -> Option<ValueRef> {
        let n = self.code[v.as_usize()].next;
        (n != LINK_NONE).then(|| ValueRef::new(n))
    }

    #[inline]
    pub fn prev_of(&self, v: ValueRef) -> Option<ValueRef> {
        let p = self.code[v.as_usize()].prev;
        (p != LINK_NONE).then(|| ValueRef::new(p))
    }

    #[inline]
    pub fn get(&self, v: ValueRef) -> &Armlet {
        &self.code[v.as_usize()]
    }

    #[inline]
    pub fn get_mut(&mut self, v: ValueRef) -> &mut Armlet {
        &mut self.code[v.as_usize()]
    }

    pub fn iter_live(&self) -> LiveIter<'_> {
        LiveIter { block: self, cursor: self.head_vr() }
    }

    pub fn iter_live_rev(&self) -> RevLiveIter<'_> {
        RevLiveIter { block: self, cursor: self.tail_vr() }
    }
}

pub struct LiveIter<'b> {
    block: &'b Block,
    cursor: Option<ValueRef>,
}

impl<'b> Iterator for LiveIter<'b> {
    type Item = (ValueRef, &'b Armlet);
    fn next(&mut self) -> Option<Self::Item> {
        let v = self.cursor?;
        let a = &self.block.code[v.as_usize()];
        self.cursor = self.block.next_of(v);
        Some((v, a))
    }
}

pub struct RevLiveIter<'b> {
    block: &'b Block,
    cursor: Option<ValueRef>,
}

impl<'b> Iterator for RevLiveIter<'b> {
    type Item = (ValueRef, &'b Armlet);
    fn next(&mut self) -> Option<Self::Item> {
        let v = self.cursor?;
        let a = &self.block.code[v.as_usize()];
        self.cursor = self.block.prev_of(v);
        Some((v, a))
    }
}
